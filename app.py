import streamlit as st
import tensorflow as tf

@st.cache(allow_output_mutation=True)
def load_model(x):
    classifier = tf.keras.models.load_model(x)
    return classifier
model = import_model('Veg.h5')
st.write("""
          # Vegetable Prediction
         """
        )
file = st.file_uploader("Please upload an vegetable image",type=['jpg'])
import cv2
from PIL import Image,ImageOps
import numpy as np
def import_and_predict(image_data,model):
    
    size = (150,150)
    image = ImageOps,fit(image_data,size,Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)
    
    return prediction
if file is None:
    st.text("Please Upload an image file")
else:
    image = Image.open(file)
    st.image(image,use_column,width=True)
    predictions = import_and_predict(image,model)
    class_names = ['Bean','Bitter_Gourd','Bottle_Gourd','Brinjal','Broccoli','Cabbage','Capsicum','Carrot','Cauliflower','Cucumber','Papaya','Potato','Pumpkin','Radish','Tomato']
    string = "This image most likely is : "+ class_names[np.argmax(predictions)]
    st.success(string)
