"""CPU-friendly cached-feature neural training, optional end-to-end fine-tuning."""
import argparse, json, random, time, copy, os, gzip
os.environ.setdefault('MPLCONFIGDIR','/tmp/plant-mpl')
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import build_model, transform

class Leaves(Dataset):
    def __init__(self,root,rows,classes,augment=False): self.root=root; self.rows=rows; self.classes=classes; self.tf=transform(augment)
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        with Image.open(self.root/r['path']) as im: x=self.tf(ImageOps.exif_transpose(im).convert('RGB'))
        return x,self.classes.index(r['label'])

def main(a):
    start=time.time(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.set_num_threads(a.threads)
    device=torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    manifest=json.loads(gzip.open(a.manifest,'rt').read() if a.manifest.suffix=='.gz' else a.manifest.read_text()); classes=manifest['classes']; rows=manifest['rows']
    splits={s:[r for r in rows if r['split']==s] for s in ['train','validation','test']}
    for s in splits:
        for other in splits:
            if s!=other: assert not ({r['group'] for r in splits[s]} & {r['group'] for r in splits[other]})
    model=build_model(len(classes),pretrained=True).to(device)
    loaders={s:DataLoader(Leaves(a.data,r,classes),batch_size=a.batch_size,num_workers=a.workers) for s,r in splits.items()}
    features={}
    for s,loader in loaders.items():
        xs=[]; ys=[]; model.eval()
        with torch.inference_mode():
            for i,(x,y) in enumerate(loader):
                f=model.avgpool(model.features(x.to(device))).flatten(1).cpu(); xs.append(f); ys.append(y)
                if i%30==0: print(f'features {s} {i}/{len(loader)}',flush=True)
        features[s]=(torch.cat(xs),torch.cat(ys))
    history=[]; best=-1; chosen=None; best_epoch=0
    train=DataLoader(TensorDataset(*features['train']),batch_size=256,shuffle=True)
    opt=torch.optim.AdamW(model.classifier.parameters(),lr=.001,weight_decay=.01)
    loss_fn=nn.CrossEntropyLoss()
    for epoch in range(a.epochs):
        model.classifier.train(); loss_sum=0; correct=0; count=0
        for x,y in train:
            x=x.to(device); y=y.to(device); opt.zero_grad(); z=model.classifier(x); loss=loss_fn(z,y); loss.backward(); opt.step()
            loss_sum+=loss.item()*len(y); correct+=(z.argmax(1)==y).sum().item(); count+=len(y)
        model.eval()
        with torch.inference_mode():
            vx,vy=features['validation']; z=model.classifier(vx.to(device)); val=accuracy_score(vy.numpy(),z.argmax(1).cpu().numpy()); vl=loss_fn(z,vy.to(device)).item()
        history.append({'stage':'head','epoch':epoch+1,'loss':loss_sum/count,'accuracy':correct/count,'val_loss':vl,'val_accuracy':val})
        if val>best: best=val; chosen=copy.deepcopy(model.state_dict()); best_epoch=epoch+1
        if epoch%5==0: print(history[-1],flush=True)
        if epoch+1-best_epoch>=15: break
    model.load_state_dict(chosen)
    # Fine-tune only after validation-selected head training; test remains untouched.
    if a.finetune_epochs:
        train_images=DataLoader(Leaves(a.data,splits['train'],classes,True),batch_size=a.batch_size,shuffle=True,num_workers=a.workers)
        opt=torch.optim.AdamW([{'params':model.features.parameters(),'lr':1e-5},{'params':model.classifier.parameters(),'lr':1e-4}],weight_decay=.01)
        for epoch in range(a.finetune_epochs):
            model.train(); total=0; correct=0; loss_sum=0
            for x,y in train_images:
                x=x.to(device);y=y.to(device);opt.zero_grad();z=model(x);loss=loss_fn(z,y);loss.backward();opt.step();total+=len(y);correct+=(z.argmax(1)==y).sum().item();loss_sum+=loss.item()*len(y)
            model.eval(); pred=[]; actual=[]
            with torch.inference_mode():
                for x,y in loaders['validation']: pred.extend(model(x.to(device)).argmax(1).cpu().tolist());actual.extend(y.tolist())
            val=accuracy_score(actual,pred); history.append({'stage':'finetune','epoch':epoch+1,'accuracy':correct/total,'loss':loss_sum/total,'val_accuracy':val})
            print(history[-1],flush=True)
            if val>best: best=val;chosen=copy.deepcopy(model.state_dict());best_epoch=epoch+1
        model.load_state_dict(chosen)
    model.eval(); actual=[];pred=[]
    with torch.inference_mode():
        for x,y in loaders['test']: pred.extend(model(x.to(device)).argmax(1).cpu().tolist());actual.extend(y.tolist())
    precision,recall,f1,_=precision_recall_fscore_support(actual,pred,average='macro',zero_division=0)
    a.out.mkdir(parents=True,exist_ok=True); Path('models').mkdir(exist_ok=True)
    torch.save({'state_dict':{k:v.cpu() for k,v in model.state_dict().items()},'classes':classes,'architecture':'mobilenet_v3_small','image_size':224},'models/plant_model.pt')
    metrics={'test_accuracy':accuracy_score(actual,pred),'macro_precision':precision,'macro_recall':recall,'macro_f1':f1,'validation_accuracy':best,'selected_epoch':best_epoch,'split_counts':{s:len(r) for s,r in splits.items()},'classes':classes,'device':str(device),'seed':a.seed,'head_epochs_run':sum(h['stage']=='head' for h in history),'finetune_epochs_run':a.finetune_epochs,'duration_seconds':round(time.time()-start),'classification_report':classification_report(actual,pred,target_names=classes,output_dict=True,zero_division=0),'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}}
    (a.out/'metrics.json').write_text(json.dumps(metrics,indent=2));(a.out/'history.json').write_text(json.dumps(history,indent=2))
    (a.out/'test_predictions.json').write_text(json.dumps([{'path':r['path'],'actual':classes[y],'predicted':classes[p]} for r,y,p in zip(splits['test'],actual,pred)],indent=2))
    labels=[c.split('___')[-1].replace('_',' ') for c in classes]
    fig,ax=plt.subplots(figsize=(12,10)); ConfusionMatrixDisplay(confusion_matrix(actual,pred),display_labels=labels).plot(ax=ax,xticks_rotation=90,colorbar=False);fig.tight_layout();fig.savefig(a.out/'confusion_matrix.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots();ax.plot([h['accuracy'] for h in history],label='train');ax.plot([h['val_accuracy'] for h in history],label='validation');ax.set(xlabel='Epoch index',ylabel='Accuracy');ax.legend();fig.tight_layout();fig.savefig(a.out/'training_curve.png',dpi=140);plt.close(fig)
    print(json.dumps({k:v for k,v in metrics.items() if k not in ['classification_report','config','classes']},indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--manifest',type=Path,default=Path('reports/manifest.json'));p.add_argument('--out',type=Path,default=Path('reports'));p.add_argument('--epochs',type=int,default=100);p.add_argument('--finetune-epochs',type=int,default=0);p.add_argument('--batch-size',type=int,default=64);p.add_argument('--workers',type=int,default=0);p.add_argument('--threads',type=int,default=8);p.add_argument('--seed',type=int,default=42);main(p.parse_args())
