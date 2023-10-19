from __future__ import division   # impone aritmética no entera en la división
from PIL import Image             # Image Loading
import numpy as np                # Arrays, matrix
import pandas as pd               # Dataframes
import matplotlib.pyplot as plt   # Graphic representation
import cv2                        # Image transformation and matrix
import os                         # Working with the terminal
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, precision_score
import tensorflow as tf           # For building the CNN
import keras                      # For building the CNN
import tensorflow.image as tfi    # Wotking with images in a TF adapted format
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D,  MaxPool2D, UpSampling2D, concatenate, Activation
from sklearn.model_selection import train_test_split # Splitting the data


import warnings
warnings.filterwarnings("ignore")


gpus = tf.config.experimental.list_physical_devices('GPU')


#GPU growth, limiting memory to the minimun
#Preventing the OOM error
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu , True)

path = "Dataset_BUSI_with_GT/"

#Create clases 
types = []

for i in os.listdir(path):
    if i != '.DS_Store':
        types.append(i)

#Separate the directories, from masks and images
mask = []
image = []

for folder in types:
    i = os.path.join(path, folder)
    
    for pic in os.listdir(i):

        if "_mask" in pic:
            mask.append(os.path.join(i, pic))
            
        else:
            image.append(os.path.join(i, pic))


def rescale(picture):
    """ 
    Re-size the vectorize image/mask of the ultrasound to 250 x 250, 
    Rescale without padding, this can reduce the model accuracy
    """
    
    return tfi.resize(cv2.imread(picture), size= (256, 256)).numpy().astype(int)

df_img = pd.DataFrame(image, columns=['image'])
df_mask = pd.DataFrame(mask, columns=['mask'])

#Dataframes
df_mask = pd.DataFrame(df_mask['mask'].sort_values())
df_img = pd.DataFrame(df_img['image'].sort_values().reset_index(drop=True))

#Remove duplicates
mask_dup = []
for i in df_mask['mask']:
    if i[-5] != '1' and i[-5] != '2':
        mask_dup.append(i)
    else:
        print(i)


X = [rescale(i) for i in df_img['image']]
X = np.array(X)

y = pd.DataFrame(mask_dup, columns=['mask'])
y = [rescale(x) for x in (y['mask'])]
y = np.array(y)


#Train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.20)

#Adjustment
X_train_grey = tf.image.rgb_to_grayscale(X_train) / 254
X_test_grey = tf.image.rgb_to_grayscale(X_test) / 254

y_train_grey = tf.image.rgb_to_grayscale(y_train) / 254
y_test_grey = tf.image.rgb_to_grayscale(y_test) / 254

#CNN Model
#Classification Model
model_classification = tf.keras.Sequential([
  keras.layers.Rescaling(1./255, input_shape=(256, 256, 1)),
  keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
  keras.layers.MaxPooling2D(),
  keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
  keras.layers.MaxPooling2D(),
  keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
  keras.layers.MaxPooling2D(),
  keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
  keras.layers.MaxPooling2D(),
  keras.layers.Dropout(0.5),
  keras.layers.Flatten(),
  keras.layers.Dense(256, activation='relu'),
  keras.layers.Dense(3,activation="softmax")
])


# Segmentation model
model_segmentation = keras.models.Sequential([
    
    #image size = 256 * 256 * 1
    keras.layers.Conv2D(16, 1, padding='same', activation='relu', input_shape=(256, 256, 1)),
    keras.layers.Conv2D(16, 1, padding='same', activation='relu', input_shape=(256, 256, 1)),
    keras.layers.MaxPooling2D(),
    keras.layers.BatchNormalization(),
    
    keras.layers.Conv2D(32, 1, padding='same', activation='relu'),
    keras.layers.Conv2D(32, 1, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    
    keras.layers.Conv2D(64, 1, padding='same', activation='relu'),
    keras.layers.Conv2D(64, 1, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.5),
    
    keras.layers.Conv2D(128, 1, padding='same', activation='relu'),
    keras.layers.Conv2D(128, 1, padding='same', activation='relu'),
    keras.layers.UpSampling2D(),
    keras.layers.BatchNormalization(),
    
    keras.layers.Conv2D(64, 1, padding='same', activation='relu'),
    keras.layers.Conv2D(64, 1, padding='same', activation='relu'),
    keras.layers.UpSampling2D(),
    
    keras.layers.Conv2D(32, 1, padding='same', activation='relu'),
    keras.layers.Conv2D(32, 1, padding='same', activation='relu'),
    keras.layers.UpSampling2D(),
    keras.layers.Dropout(0.5),
    
    keras.layers.Conv2D(16, 1, padding='same', activation='relu'),
    keras.layers.Conv2D(16, 1, padding='same', activation='relu'),
    keras.layers.Conv2D(1, 1, activation='sigmoid')

])


train_generator = tf.keras.preprocessing.image_dataset_from_directory(
    path,
    image_size=(256, 256),
    batch_size=32,
    seed=77,
    color_mode='grayscale', 
    subset='training',
    shuffle=True,
    validation_split=0.2
)

val_generator = tf.keras.preprocessing.image_dataset_from_directory(
    path,
    image_size=(256, 256),
    seed=77,
    batch_size=32,
    color_mode='grayscale',
    subset='validation',
    shuffle=True,
    validation_split=0.2)

#To save all check points from the 
callbacks = [ keras.callbacks.ModelCheckpoint("image_segmentation.h5", save_best_only=True) ]
callbacks_2 = [ keras.callbacks.ModelCheckpoint("image_classification2.h5", save_best_only=True) ]

model_segmentation.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model_classification.compile(optimizer="Adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"])


#Segmentation fit
model_segmentation.fit(X_train_grey, y_train_grey, epochs=25, batch_size=30, validation_data=(X_test_grey, y_test_grey), callbacks=callbacks)

#Classification fit
model_classification.fit(train_generator, epochs=20, validation_data=val_generator, callbacks=callbacks_2)

#Segmentation predict
y_pred_segmentation = model_segmentation.predict(X_test_grey)

#Classification rpedict
y_pred_classification = model_classification.predict(X_test_grey)