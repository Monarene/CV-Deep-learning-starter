# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 15:01:06 2018

@author: Michael
"""

#importing the neccessary libraries
#importing the relevant libraries
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Flatten, Activation
from keras.models import Sequential
from sklearn import datasets
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras import backend as K
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import numpy as np

#importing the relevant datasets
print("[INFO] importing the dataset")
dataset = datasets.fetch_mldata("MNIST Original")
data = dataset.data
if K.image_data_format() == "channels_first":
    data = data.reshape(data.shape[0],1,28,28)
else:
    data = data.reshape(data.shape[0], 28, 28, 1)
trainX, testX, trainY, testY = train_test_split(data.astype('float')/255.0, dataset.target, random_state=42,
                                                test_size=0.25)
lb= LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

#building the network and testing it
sgd = SGD(lr=0.01)

#buildiing the CNN architecture
model = Sequential()
model.add(Conv2D(20,(5,5),padding="same", input_shape=(28,28,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(50,(5,5), padding="same"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())        
model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer=sgd,metrics=['accuracy'])
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=20, batch_size=128)














    