# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 15:19:41 2018

@author: Michael
"""

#importing relevant libraries
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from utilities.preprocessing import SimplePreprocessor, ImageToArrayPreprocessor
from utilities.datasets import SimpleDatasetLoader
from utilities.nn.cnn import ShallowNet
from keras.optimizers import SGD
from keras.models import load_model
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import cv2

#miporting the dataset and preprocessing 
imagePaths = list(paths.list_images(r'C:\Users\Michael\Desktop\Data Science\DL4CVStarterBundle-master\DL4CVStarterBundle-master\datasets\animals'))
sp = SimplePreprocessor(32,32)
ita = ImageToArrayPreprocessor()
sdl = SimpleDatasetLoader(preprocessors=[sp])
(dataset, labels) = sdl.load(imagePaths, verbose=500)
dataset = dataset.astype("float")/255.0

trainX, testX, trainY, testY = train_test_split(dataset, labels, test_size=0.25,
                                                random_state=42)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

#training the network
print("[INFO] Uploading model")
sgd = SGD(lr = 0.005)
model = ShallowNet.build(32,32,3,3)
model.compile(optimizer=sgd, loss = "categorical_crossentropy", metrics=['accuracy'])
H = model.fit(trainX, trainY,validation_data =(testX, testY), epochs = 100 , batch_size =32)

#evaluating the network
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), 
                            target_names=lb.classes_))


plt.figure()
plt.style.use('ggplot')
plt.plot(np.arange(0,100), H.history["loss"], label = "Training Loss")
plt.plot(np.arange(0,100), H.history["val_loss"], label = "Validation Loss")
plt.plot(np.arange(0,100), H.history["acc"], label = "Training accuray")
plt.plot(np.arange(0,100), H.history["val_acc"], label = "Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

#serializing the model
model.save("shallownet_animals.hdf5")

#load the model from disk and preprocesss 10 random loaded images
saved_model = load_model("shallownet_animals.hdf5")
imagePaths = np.array(imagePaths)
indexes = np.random.randint(0, len(imagePaths), size=(10,))
thePics = imagePaths[indexes]

sp = SimplePreprocessor(32,32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(thePics, verbose=500)

#predicting and draing
preds = saved_model.predict(data.astype("float")/255.0, batch_size=32).argmax(axis=1)
classLabels = ["cat", "dog", "panda"]
for (i, path) in enumerate(thePics):
    image = cv2.imread(path)
    if classLabels[preds[i]] == "cat":
        cv2.putText(image, "Label : Cat", (10,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0),2)
    elif classLabels[preds[i]] == "dog":
        cv2.putText(image, "Label : dog", (10,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
    else:
        cv2.putText(image, "Label : panda", (10,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)























