# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 17:22:59 2018

@author: Michael
"""
#importing libraries
from sklearn.preprocessing import LabelBinarizer
from keras.datasets import cifar10
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from utilities.nn.cnn import ShallowNet
import matplotlib.pyplot as plt
import numpy as np

#imprting the dataset and preprocessing
print("[INFO] importing dataset")
(trainX, trainY), (testX, testY) = cifar10.load_data()
trainX = trainX.astype("float")/255.0
testX = testX.astype("float")/255
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

#building model and deploying it 
label_names = ["airplane", "automobile","bird", "cat","deer","dog",
               "frog","horse","ship","truck"]
sgd = SGD(lr=0.01)
model = ShallowNet.build(32,32,3,10)
model.compile(optimizer=sgd,loss="categorical_crossentropy",metrics=['accuracy'])
model.fit(trainX, trainY, validation_data = (testX, testY), epochs=100, batch_size=32,verbose=1)

#evaluating the results





























