import os
import sys
import glob
import random
import warnings
import itertools
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import cv2
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from IPython.display import SVG
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, ElementTree
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10, cifar100
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Conv2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

CIFAR100_CLASSES = sorted(['beaver', 'dolphin', 'otter', 'seal', 'whale',  # aquatic mammals
                           'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',  # fish
                           'orchids', 'poppies', 'roses', 'sunflowers', 'tulips', # flowers
                           'bottles', 'bowls', 'cans', 'cups', 'plates', # food containers
                           'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers', # fruit and vegetables
                           'clock', 'computer' 'keyboard', 'lamp', 'telephone', 'television', # household electrical devices
                           'bed', 'chair', 'couch', 'table', 'wardrobe', # household furniture
                           'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach', # insects
                           'bear', 'leopard', 'lion', 'tiger', 'wolf', # large carnivores
                           'bridge', 'castle', 'house', 'road', 'skyscraper', # large man-made outdoor things
                           'cloud', 'forest', 'mountain', 'plain', 'sea', # large natural outdoor scenes
                           'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo', # large omnivores and herbivores
                           'fox', 'porcupine', 'possum', 'raccoon', 'skunk', # medium-sized mammals
                           'crab', 'lobster', 'snail', 'spider', 'worm', # non-insect invertebrates
                           'baby', 'boy', 'girl', 'man', 'woman', # people
                           'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle', # reptiles
                           'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel', # small mammals
                           'maple', 'oak', 'palm', 'pine', 'willow', # trees
                           'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train', # vehicles 1
                           'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor' # vehicles 2
                          ])

# loading CIFAR-100 dataset
(X_train, y_train), (X_test, y_test) = cifar100.load_data()
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, shuffle=True, random_state=0)

# reshaping image data
X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
X_valid = X_valid.reshape(X_valid.shape[0], 32, 32, 3)
X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)

# converting integer to float
X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_test = X_test.astype('float32')

# data normalization
X_train = X_train / 255
X_valid = X_valid / 255
X_test = X_test / 255

# one hot encoding
y_train = to_categorical(y_train, 100)
y_valid = to_categorical(y_valid, 100)
y_test = to_categorical(y_test, 100)

# input shape for the model
INPUT_SHAPE = (32, 32, 3)

# Creating a Sequential model
model_cnn = Sequential()

# Adding Convolutional layers
model_cnn.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3),padding='same'))

model_cnn.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu',padding='same'))
model_cnn.add(MaxPooling2D(pool_size=(2, 2)))

model_cnn.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu',padding='same'))
model_cnn.add(MaxPooling2D(pool_size=(2, 2)))

model_cnn.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu',padding='same'))
model_cnn.add(MaxPooling2D(pool_size=(2, 2)))
model_cnn.add(Dropout(0.25))

model_cnn.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu',padding='same'))
model_cnn.add(MaxPooling2D(pool_size=(2, 2)))

model_cnn.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu',padding='same'))
model_cnn.add(MaxPooling2D(pool_size=(2, 2)))
model_cnn.add(Dropout(0.25))

model_cnn.add(Flatten())

model_cnn.add(Dense(1024, activation='relu'))
model_cnn.add(Dropout(0.5))

model_cnn.add(Dense(len(CIFAR100_CLASSES), activation='softmax'))


datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

datagen.fit(X_train)

# Compiling the model with Adam optimizer and categorical crossentropy loss
model_cnn.compile(loss=categorical_crossentropy,
                  optimizer=Adam(learning_rate=0.0001),
                  metrics=['acc', 'top_k_categorical_accuracy'])

EPOCHS = 100
BATCH_SIZE = 32
early_stopping = EarlyStopping(monitor='val_loss',mode='min', patience=10, restore_best_weights=True)


# Training the model
history_cnn = model_cnn.fit(datagen.flow(X_train,
                            y_train,batch_size=BATCH_SIZE),
                            steps_per_epoch=X_train.shape[0] // 32,
                            epochs=EPOCHS,
                            batch_size=BATCH_SIZE,
                            validation_data=(X_valid, y_valid),
                            callbacks=[early_stopping])

# Evaluating the model on the test set
test_loss_cnn, test_acc_cnn, test_top5_acc_cnn = model_cnn.evaluate(X_test, y_test, verbose=0)
print('CNN Test Accuracy: {}'.format(test_acc_cnn))

# Plotting training and validation loss curves
loss_cnn = history_cnn.history['loss']
val_loss_cnn = history_cnn.history['val_loss']
epochs_cnn = range(1, len(loss_cnn) + 1)

print('CNN epochs : {}'.format(epochs_cnn))
print('CNN Training loss : {}'.format(loss_cnn))
print('CNN Validation loss : {}'.format(val_loss_cnn))
