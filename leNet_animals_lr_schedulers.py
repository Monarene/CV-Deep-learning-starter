# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 17:59:22 2019

@author: Michael
"""

#Importing the neccesary libraries
import matplotlib
matplotlib.use("Agg")
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utilities.nn.cnn import LeNet
from utilities.nn.cnn import MiniVGGNet
from utilities.datasets import SimpleDatasetLoader
from utilities.preprocessing import SimplePreprocessor
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.utils import plot_model
import pydot
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from ann_visualizer.visualize import ann_viz;

#importing and preprocessing the dataset
imagePaths = list(paths.list_images(r'C:\Users\Michael\Desktop\Data Science\DL4CVStarterBundle-master\DL4CVStarterBundle-master\datasets\animals'))
sp = SimplePreprocessor(32,32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(dataset, labels) = sdl.load(imagePaths, verbose=1000)
dataset = dataset.astype('float')/255.0
trainX, testX, trainY, testY = train_test_split(dataset, labels, random_state=42,
                                                test_size = 0.25)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

#build the ML model and use Learning_rate _decay
sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
#model = LeNet.build(width=32, height=32, depth=3, classes = 3)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes = 3)
model.compile(optimizer=sgd, loss = "categorical_crossentropy", metrics=['accuracy'])
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=30, batch_size=64)

#evaluating 
plt.figure("Training MiniVGGNet on animals without Decay")
plt.style.use('ggplot')
plt.plot(np.arange(0,30), H.history['acc'], label = "Training accuracy")
plt.plot(np.arange(0,30), H.history['val_acc'], label = "Validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("Training MiniVGGNet without decay on animals dataset ")

#trainging the MiniVGGnet this time around with decay
sgd_1 = SGD(lr=0.01, momentum=0.9, nesterov=True, decay = 0.01/30)
model_1 = MiniVGGNet.build(width = 32, height = 32, depth = 3, classes=3)
model_1.compile(optimizer=sgd_1, loss = "categorical_crossentropy", metrics=['accuracy'])
H_1 = model_1.fit(trainX, trainY, validation_data=(testX, testY), epochs=30, batch_size=64)

#evaluating the model_1
plt.figure("Training MiniVGGNet on animals with Decay")
plt.style.use('ggplot')
plt.plot(np.arange(0,30), H_1.history['acc'], label = "Training accuracy")
plt.plot(np.arange(0,30), H_1.history['val_acc'], label = "Validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("Training MiniVGGNet with decay on animals dataset ")

#defining the relevant directories
def step_decay(epoch):
    initAlpha = 0.01
    factor = 0.5
    dropEvery = 10
    alpha = initAlpha * (factor**np.floor((1 + epoch)/ dropEvery))
    return float(alpha)

callbacks = [LearningRateScheduler(step_decay)]
model_2 = MiniVGGNet.build(width = 32, height = 32, depth = 3, classes=3)
model_2.compile(optimizer=sgd, loss = "categorical_crossentropy", metrics=['accuracy'])
H_2 = model_2.fit(trainX, trainY, validation_data=(testX, testY), epochs=30, batch_size=64)

#evaluating the model_1
plt.figure("Training MiniVGGNet on animals with LRS = 0.5")
plt.style.use('ggplot')
plt.plot(np.arange(0,30), H_2.history['acc'], label = "Training accuracy")
plt.plot(np.arange(0,30), H_2.history['val_acc'], label = "Validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("Training MiniVGGNet with LSR ")

#this timwe arpound using a model xheckpointing callback
checkpoint = ModelCheckpoint("Best minivggnet from animals.hdf5", monitor="val_acc",
                             save_best_only=True, verbose=1)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes = 3)
model.compile(optimizer=sgd_1, loss = "categorical_crossentropy", metrics=['accuracy'])
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=30,
              callbacks=[checkpoint],batch_size=64)

plot_model(model, to_file="minivggnet on animals.png", show_shapes=True)
ann_viz(model, title="minivggnet")
























