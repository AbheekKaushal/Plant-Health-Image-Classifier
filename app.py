from pathlib import Path
import streamlit as st
from PIL import Image, UnidentifiedImageError
from core import load_model, predict
st.set_page_config(page_title='Plant Health Classifier',page_icon='🌿')
st.title('🌿 Plant Health Image Classifier')
st.write('Upload a clear photo of one tomato leaf. The model recognises ten healthy/disease classes.')
st.caption('Trained on controlled PlantVillage images. Field accuracy is unverified; confidence is not a calibrated probability of diagnosis.')
@st.cache_resource
def get_model(): return load_model('models/plant_model.pt')
if not Path('models/plant_model.pt').exists():
    st.info('Train the model first using the README instructions.');st.stop()
model,classes=get_model()
file=st.file_uploader('Tomato leaf photo',type=['jpg','jpeg','png'])
if file:
    try:
        im=Image.open(file);im.load();st.image(im,width=350)
        results=predict(model,classes,im)
        label,score=results[0];st.subheader(label.split('___')[-1].replace('_',' '));st.metric('Model confidence',f'{score:.1%}')
        for label,score in results[:3]: st.write(f"{label.split('___')[-1].replace('_',' ')}: {score:.1%}")
    except (UnidentifiedImageError,OSError):st.error('Please upload a valid image.')
