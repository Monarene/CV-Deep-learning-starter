# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 18:22:59 2018

@author: Michael
"""
#importing the necessary libraries
import keras
from keras.datasets import cifar10
from keras import backend as K
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import matplotlib
import matplotlib.pyplot as plt
from keras.optimizers import SGD
matplotlib.use("Agg")
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K

#importing the necessary datasets
(trainX, trainY), (testX, testY) = cifar10.load_data()
trainX = trainX.astype("float")/255.0
testX = testX.astype("float")/255.0
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

#build the architecture
channel_dim = -1
model = Sequential()
model.add(Conv2D(32,(3,3), padding="same",input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(BatchNormalization(axis=channel_dim))
model.add(Conv2D(32,(3,3), padding="same"))
model.add(Activation('relu'))
model.add(BatchNormalization(axis=channel_dim))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3), padding="same"))
model.add(Activation('relu'))
model.add(BatchNormalization(axis=channel_dim))
model.add(Conv2D(64,(3,3), padding="same"))
model.add(Activation('relu'))
model.add(BatchNormalization(axis=channel_dim))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization(axis=channel_dim))
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation('softmax'))
#importing the DL algorithm
sgd = SGD(lr=0.01, decay=0.01/40, momentum=0.9,nesterov=True)
model.compile(optimizer = sgd, loss="categorical_crossentropy", metrics=['accuracy'])
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=40, batch_size=64)








