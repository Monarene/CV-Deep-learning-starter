# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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

#doinig all the argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required= True,
                help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
                help="# of nearest neighbors in classification")
ap.add_argument("-j","--jobs", type=int, default=-1,
                help="# of default jobs for k nearest neighbors ")
args = vars(ap.parse_args())

print("[INFO ]Loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
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

#training and fitting the model
classifier = KNeighborsClassifier(n_neighbors=2,
                                  n_jobs = -1)
classifier.fit(trainX, trainY)
print(classification_report(testY,
                            classifier.predict(testX),target_names = le.classes_))

print(confusion_matrix(testY, classifier.predict(testX)))





























