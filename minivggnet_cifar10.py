# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 17:00:56 2018

@author: Michael
"""

#importing the relevant libraries
from keras.datasets import cifar10
from keras import backend as K
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from utilities.nn.cnn import MiniVGGNet
from keras.optimizers import SGD
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

#importing the dataset and preprocessing
(trainX, trainY), (testX, testY) = cifar10.load_data()
trainX = trainX.astype("float")/255.0
testX = testX.astype("float")/255.0
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

#importing the DL algorithm
sgd = SGD(lr=0.01, decay=0.01/40, momentum=0.9,nesterov=True)
model = MiniVGGNet().build(width=32, height=32, depth=3, classes = 10)
model.compile(optimizer = sgd, loss="categorical_crossentropy", metrics=['accuracy'])
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=40, batch_size=64)












