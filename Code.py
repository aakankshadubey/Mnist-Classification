#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 11:12:22 2020
Mnist Classification - Convolutional Neural Network
@author: Aakanksha Dubey
"""


# Import Libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# Import Dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()


# Visualize Image
plt.imshow(X_train[0], cmap='Greys')


# Reshape Array
X_train.shape
X_train.shape[0]

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)



# Scale Dataset
X_train = X_train.astype("float32")
X_train = X_train/255

X_test = X_test.astype("float32")
X_test = X_test/255



# Model Outline

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


#Initialising CNN

model = Sequential()

# First Layer
input_shape = (28, 28, 1)

model.add(Convolution2D(32, kernel_size = (3, 3), 
                        input_shape = input_shape, 
                        activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

# Second Layer
model.add(Convolution2D(32, kernel_size = (3, 3), 
                        activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))


# Flattening
model.add(Flatten())


# Complete Connection
model.add(Dense(output_dim = 128, activation = 'relu'))
model.add(Dense(output_dim = 10, activation = 'sigmoid'))


# Compile CNN
model.compile(optimizer = 'adam', 
              loss = 'sparse_categorical_crossentropy', 
              metrics = ['accuracy'])



# Fitting Model to Dataset
model.fit(X_train, y_train, epochs = 10)


# Evaluate Model

model.evaluate(X_test, y_test)
accuracy = model.evaluate(X_test, y_test)[1]
accuracy
model.summary()


# Check on a Random Image

X_test.shape
img = X_test[1051]
plt.imshow(img.reshape(28, 28),cmap='Greys')

pred = model.predict_classes(img.reshape((1, 28, 28, 1)))
pred
result = pred[0]
print(result)





