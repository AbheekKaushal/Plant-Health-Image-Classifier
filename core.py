"""Shared architecture and preprocessing for training and inference."""
import torch
from torch import nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision import transforms
from PIL import ImageOps

def build_model(classes, pretrained=False):
    model=mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
    model.classifier=nn.Sequential(nn.Linear(576,256),nn.Hardswish(),nn.Dropout(.2),nn.Linear(256,classes))
    return model

def transform(augment=False):
    steps=[transforms.Resize(256), transforms.CenterCrop(224)]
    if augment: steps=[transforms.RandomResizedCrop(224,scale=(.75,1.)),transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),transforms.RandomRotation(20),transforms.ColorJitter(.15,.15,.1,.02)]
    return transforms.Compose(steps+[transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])

def load_model(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    model=build_model(len(checkpoint['classes'])); model.load_state_dict(checkpoint['state_dict']); model.eval()
    return model,checkpoint['classes']

def predict(model,classes,image):
    x=transform()(ImageOps.exif_transpose(image).convert('RGB')).unsqueeze(0)
    with torch.inference_mode(): probs=model(x).softmax(1)[0]
    return sorted(zip(classes,probs.tolist()),key=lambda r:r[1],reverse=True)
