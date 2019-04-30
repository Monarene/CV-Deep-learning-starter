# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 17:35:33 2019

@author: Michael
"""

#importing necessary libraries
from keras.applications import ResNet50, Xception, InceptionV3
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.utils import plot_model
from imutils import paths
from utilities.preprocessing import SimplePreprocessor
from utilities.datasets import SimpleDatasetLoader
import graphviz
from ann_visualizer.visualize import ann_viz;



#building the inference guy
MODELS = {"resnet50":ResNet50, "xception":Xception, "inception": InceptionV3}
model_name = input("Please input name of pretrained ImageNet model to use: ")
if model_name not in MODELS.keys():
    raise AssertionError("The model name should either be xception, resnet50, inception")

input_shape = (224, 224)
preprocess = imagenet_utils.preprocess_input
if model_name in ("xception", "inception"):
    input_shape = (299, 299)
    preprocess = preprocess_input
(height, width) = input_shape

Network = MODELS[model_name]
model = Network(weights="imagenet")

#importing the dataset and exploring its possibilities
imagePaths = list(paths.list_images(r"C:\Users\Michael\Desktop\Data Science\DL4CVStarterBundle-master\DL4CVStarterBundle-master\datasets\animals"))
sp =SimplePreprocessor(height, width)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(dataset, _) = sdl.load(imagePaths)
images = preprocess(dataset)
preds = model.predict(dataset) 
P = imagenet_utils.decode_predictions(preds)


model = Xception()
plot_model(model, to_file="Xception.png", show_shapes=True)
ann_viz(model,title="some neural network model", filename="picture.gv")





