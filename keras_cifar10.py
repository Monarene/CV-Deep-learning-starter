# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 19:13:23 2018

@author: Michael
"""
#importing the neccesary libraries
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

#importing the deep learning modules
import keras
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.core import Dense
from keras.datasets import cifar10

#importing the dataset and preprocssing
(trainX, trainY), (testX, testY) = cifar10.load_data()
trainX = trainX.reshape((trainX.shape[0], 3072))
testX = testX.reshape((testX.shape[0], 3072))
trainX = trainX.astype("float")/255.0
testX = testX.astype("float")/255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
label_names = ["airplane","automobile","bird","car","deer","dog",
               "frog","horse","ship","truck"]

sgd = SGD(0.01)
#building the network
model = Sequential()
model.add(Dense(1024, input_shape = (3072,), activation = "relu"))
model.add(Dense(512, activation = "relu"))
model.add(Dense(10, activation = "softmax"))
model.compile(loss="categorical_crossentropy", optimizer = sgd, metrics = ['accuracy'])
H = model.fit(trainX, trainY, validation_data = (testX, testY), epochs = 100,
          batch_size = 32)

#evaluating the network and visualizing the results
print("[INFO] Evaluating the network model")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),target_names=label_names))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,100), H.history["loss"], label= "Training data loss")
plt.plot(np.arange(0,100), H.history['val_loss'], label = "Test loss")
plt.plot(np.arange(0,100), H.history['acc'], label = "Training data accuracy")
plt.plot(np.arange(0,100), H.history["val_acc"], label = "Validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss / Accuracy")
plt.legend()
plt.title("Training Loss and Accuracy vs Epochs")



















