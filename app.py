
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model('best_model.h5')

IMG_SIZE = (150, 150)

st.title("Pneumonia Detection from Chest X-Ray")
st.write("Upload a chest X-ray image to detect if it's NORMAL or PNEUMONIA.")

uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Chest X-ray.', use_container_width=True)

    image = image.resize(IMG_SIZE)
    img_array = np.array(image)

    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)

    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0) 

    prediction = model.predict(img_array)[0][0]

    if prediction < 0.5:
        st.success(f"Prediction: **NORMAL** (Confidence: {1 - prediction:.2f})")
    else:
        st.error(f"Prediction: **PNEUMONIA** (Confidence: {prediction:.2f})")
