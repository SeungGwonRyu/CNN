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
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Conv2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
from hyperopt import fmin, tpe, hp
from functools import partial
from hyperopt import STATUS_OK

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

EPOCHS = 100
BATCH_SIZE = 256

from hyperopt import hp
space = {
    'num_blocks': hp.choice('num_blocks', [2, 3, 4, 5]),
    'batch_size': hp.choice('batch_size',[32,64,128,256]),
    'dropout_rate': hp.loguniform('dropout_rate',np.log(0.2),np.log(0.5)),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.001))
}

def objective (space):
    model = Sequential()
    model.add(Conv2D(128, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'))
    for i in range(space['num_blocks']):
        if i > 0:
            model.add(Conv2D(128 * 2**i, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(space['dropout_rate']))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(CIFAR100_CLASSES), activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=space['learning_rate']),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False
    )
    datagen.fit(X_train)

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(datagen.flow(X_train, y_train, batch_size=space['batch_size']),
                        validation_data=(X_valid, y_valid),
                        steps_per_epoch=X_train.shape[0] // space['batch_size'],
                        epochs=EPOCHS,
                        callbacks=[early_stop])
    best_loss = min(history.history['val_loss'])
    return {'loss': best_loss, 'status': STATUS_OK}

from hyperopt import fmin, tpe, Trials

trials = Trials()

best_hyperparams = fmin(fn=objective,
                        space=space,
                        algo=tpe.suggest,
                        max_evals=20,
                        trials=trials)

print("Best Hyperparameters:", best_hyperparams)

                        

                          
