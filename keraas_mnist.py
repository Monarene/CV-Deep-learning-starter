# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 16:32:27 2018

@author: Michael
"""

#importing the neccesary libraries
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

#importing the deep learning modules
import keras
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.core import Dense

#importing and splitting the dataset
print("[INFO] Loading the mnist dataset...")
dataset = datasets.fetch_mldata("MNIST Original")
data = dataset.data.astype("float")/255.0
trainX, testX, trainY, testY = train_test_split(data, dataset.target, test_size=0.25)

#use the LabelBinarizer to convert
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

#doing the deep leaerning
sgd =SGD(0.01)
model = Sequential()
model.add(Dense(256, input_shape = (trainX.shape[1],),activation = "sigmoid"))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(10, activation = "softmax"))
print("[INFO] doing the deep learning")
model.compile(optimizer=sgd, loss = "categorical_crossentropy", metrics=['accuracy'])
H = model.fit(trainX, trainY, validation_data = (testX, testY), epochs=100,
          batch_size=128)

#evaluating the network and visualizing the results
print("[INFO] Evaluating the network model")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1)))

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




















