import streamlit as st
import numpy as np
import tensorflow as tf 
from PIL import Image
import io
import os

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/fashion_mnist_model.h5"

model = tf.keras.models.load_model(model_path)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((28, 28)).convert('L')  # Convert to grayscale
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

st.title("Fashion MNIST Classifier")
st.write("Upload an image of a clothing item, and the model will predict its category.")
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    col1,col2 = st.columns(2)
    with col1:
        # Display the original uploaded image
        display_img = Image.open(uploaded_file)
        st.image(display_img, caption='Uploaded Image', use_column_width=True)

    with col2:
        if st.button("classify"):
            img_array = preprocess_image(uploaded_file)
            predictions = model.predict(img_array)
            predicted_class = class_names[np.argmax(predictions)]
            confidence = np.max(predictions) * 100
            st.write(f"Predicted Class: {predicted_class}")
            st.write(f"Confidence: {confidence:.2f}%")
            st.success("Classification complete!")
            st.balloons()





