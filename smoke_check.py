"""Check data isolation, saved-model predictions and the Streamlit app."""
import argparse,gzip,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from PIL import Image
from core import load_model,predict
from train import Leaves
from streamlit.testing.v1 import AppTest

def read(path):
    return json.loads(gzip.open(path,'rt').read() if path.suffix=='.gz' else path.read_text())
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(8)
    manifest=read(Path('reports/manifest.json.gz'));rows=manifest['rows']
    for key in ['group','sha256']:
        partitions=[{r[key] for r in rows if r['split']==s} for s in ['train','validation','test']]
        assert all(not partitions[i]&partitions[j] for i in range(3) for j in range(i))
    model,classes=load_model('models/plant_model.pt');expected=read(Path('reports/test_predictions.json.gz'))[:32]
    test=[r for r in rows if r['split']=='test'][:32]
    x,y=next(iter(DataLoader(Leaves(a.data,test,classes),batch_size=32)))
    with torch.inference_mode(): pred=model(x).argmax(1).tolist()
    assert [classes[i] for i in pred]==[r['predicted'] for r in expected]
    with Image.open(a.data/test[0]['path']) as im: assert abs(sum(s for _,s in predict(model,classes,im))-1)<1e-5
    app=AppTest.from_file('app.py').run(timeout=30);assert not app.exception,app.exception
    print('PASS: split isolation, saved-model predictions, probabilities, and app startup')
