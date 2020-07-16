# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 21:09:28 2018

@author: Michael
"""

#importing the relevant libraries
from sklearn import datasets
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from utilities.nn.cnn import LeNet
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
model = LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer=sgd)
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=20, batch_size=128)

#wriint the functions to evaluate the dataset
plt.figure()
plt.style.use('ggplot')
plt.plot(np.arange(0,20), H.history['loss'], label = "Training Loss")
plt.plot(np.arange(0,20), H.history['val_loss'], label = "Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()


#wriint the functions to evaluate the dataset
plt.figure()
plt.style.use('ggplot')
plt.plot(np.arange(0,20), H.history['acc'], label = "Training accuracy")
plt.plot(np.arange(0,20), H.history['val_acc'], label = "Validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()








