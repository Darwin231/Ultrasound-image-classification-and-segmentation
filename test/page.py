import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import image as tfi 
import cv2
import matplotlib.pyplot as plt
import os


# Load your pre-trained model
segmentation_model = keras.models.load_model("models/image_segmentation.h5")
classification_model = keras.models.load_model("models/image_classification2.h5")

st.set_page_config(page_title="Tumor Recognition App")

st.title("Tumor Recognition App")


image_file = st.file_uploader('Upload your image', type=["jpg", "jpeg", "png"])

#Rescale function
def rescale(picture):
    """ 
    Re-size the vectorize image/mask of the ultrasound to 250 x 250, 
    Rescale without padding, this can reduce the model accuracy
    """
    return tfi.resize(cv2.imread(picture), size= (256, 256)).numpy().astype(int)

#Apply model to the loaded image
if image_file is not None:
    if "benign" in str(image_file.name):
        path = f"../Medical image recognition/Dataset_BUSI_with_GT/benign/{image_file.name}"

    elif "malignant" in str(image_file.name):
        path = f"../Medical image recognition/Dataset_BUSI_with_GT/malignant/{image_file.name}"

    else:
        path = f"../Medical image recognition/Dataset_BUSI_with_GT/normal/{image_file.name}"
    # Load the image

    image = Image.open(image_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    

    # Preprocess the image
    image_rgb = rescale(path)
    pred_im = tfi.rgb_to_grayscale(image_rgb) / 254
    pred_im_clas = tfi.rgb_to_grayscale(image_rgb)
    
    # Make a prediction
    prediction_seg = segmentation_model.predict(np.expand_dims(pred_im, axis=0))
    predicted_class= np.argmax(classification_model.predict(np.expand_dims(pred_im_clas, axis=0)), axis=-1)[0]

    #Reshape
    pred_seg = np.reshape(prediction_seg, (256, 256, 1))
    
    if predicted_class == 0:
        predicted_class = 'Benign'
    elif predicted_class == 1:
        predicted_class = 'Malignant'
    elif predicted_class == 2:
        predicted_class = 'Normal'
    else:
        predicted_class = 'No funciona'


   
    st.header("Prediction")

    plt.imshow(pred_seg)
    plt.title(f'Etiqueta verdadera: {image_file.name}, Etiqueta predicha: {predicted_class}')
    plt.savefig('test', dpi = 400)

    st.image('test.png', caption='Segmentation Model', use_column_width=True)

    os.remove('test.png')