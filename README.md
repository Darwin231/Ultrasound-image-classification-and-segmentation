# Ultrasound-image-classification-and-segmentation

__*Disclaimer*__: *This project do not pretend to do the work of a Doctor, this project will help the analysis for a Doctor.* 🩺


## 1. Purpose

This is not a *commercial software*. The main objective of this project is to build an opensource code for medical image analysis, in this case the model was trainned with breast cancer images. In the future more cancer images will be added to help identifying cancer in early stages.

We created via ML model a classification and segmentation model for breast ultrasound cancer. Cancer is a very hard disease that in some cases cannot be cured, but an early detection will help a lot of people. ❤️

*The page.py file is an stremlit framework with a demo for the app* Nowadays we don´t received printed ultra-sound images, these images are sent directly to the doctor, the demo is a preview of the tool for the doctors.


## 2. How the project was built?

We apply Convolutional networks with Tensorflow mainly (CNN). If you are not familiar with CNN this is how they work:

![59954intro to CNN](https://user-images.githubusercontent.com/96625479/233455423-bf3f0397-26cf-49b2-9708-7c8c76616640.JPG)

Basically Convolutional network were used for idetifying the behavior of tumors on the Ultra-sound pictures a total of 780 images were used to trainned the model with thre classifications: Normal (No tumor), benign and malignant.

dataset link: https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset


### 2.1 Architecture

The dataset is composed on images only. To process the images we used images library from tensorflow to vectorize the images and then convert them to grey scale to have a better processing, input shape used for all the images was (256, 256, 1)). No padding was used for this model.


#### 2.1.1__Structure__:

- Convolutional networks
- Batch Normalization
- Dense
- Dropout
- Flatten
           

## Important info

Models are included into the models folder. Also a .Dockerfile was added to deploy on a separated environmet.

To execute the docker remember to first call the python 3.9 image with the commadn:
           docker pull python:3.9 


