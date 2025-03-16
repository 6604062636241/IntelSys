import os
import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
from tensorflow.keras.preprocessing import image

st.set_page_config(page_title="Neural Network Model", layout="wide")

gdown.download("https://drive.google.com/uc?id=1uDyqphczWGPxI-cOv_bhiQ9IyUzp_-gz", "food101_model.h5", quiet=False)

model = tf.keras.models.load_model("food101_model.h5")

model.summary()
class_names_file = "classes.txt"
with open(class_names_file, "r") as file:
    class_names = [line.strip() for line in file.readlines()]

print(class_names)

def predict_image(img):
    img = img.resize((224, 224)) 
    img_array = np.array(img) / 255.0 
    img_array = np.expand_dims(img_array, axis=0) 

    prediction = model.predict(img_array)
    print(prediction)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)  

    return predicted_class, confidence

st.title("Food Predict")

uploaded_file = st.file_uploader("Upload your picture", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption="Uploaded successfully", use_container_width=True)

    result, conf = predict_image(img)

    st.write(f"Predicted class: {result}")
    st.write(f"Confidence: {conf:.2%}")
