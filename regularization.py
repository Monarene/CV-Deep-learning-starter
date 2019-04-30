# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 10:14:11 2018

@author: Michael
"""

import utilities

#importting the guys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(42)

#importing the necessary libraries
import argparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from imutils import paths
from utilities.preprocessing import SimplePreprocessor
from utilities.datasets import SimpleDatasetLoader 
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier

print("[INFO ]Loading images...")
imagePaths = list(paths.list_images(r'C:\Users\Michael\Desktop\Data Science\DL4CVStarterBundle-master\DL4CVStarterBundle-master\datasets\animals'))

#initialize the image processor 
# load the dataset from disk
sp = SimplePreprocessor(32,32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape(data.shape[0], 3072)

print("[INFO] features matrix {:.1f}".format(data.nbytes /(1024 * 1024)))

#encode the labels 
le = LabelEncoder()
labels = le.fit_transform(labels)

#splitting the dataset
trainX, testX, trainY, testY = train_test_split(data, labels, random_state=42)

#writing the loop to train the dataset on three regularization parameters
for r in [None, 'l1','l2']:
    print("[INFO] Training with regularization '{}' penalty".format(r))
    model = SGDClassifier(loss='log', penalty=r,max_iter=10,learning_rate="constant",
                          eta0=0.01, random_state=42)
    model.fit(trainX, trainY)
    acc = model.score(testX, testY)
    print("[INFO] accuracy on '{}' regularization penalty is {:.2f}%".format(r, acc*100))




