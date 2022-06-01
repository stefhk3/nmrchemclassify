#this implements https://towardsdatascience.com/how-to-cluster-images-based-on-visual-similarity-cd6e7209fe34

# for loading/processing the images  
from tensorflow.keras.utils import array_to_img 
from tensorflow.keras.utils import img_to_array 
from keras.applications.vgg16 import preprocess_input 
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

# models 
from keras.applications.vgg16 import VGG16 
from keras.models import Model

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# for everything else
import os
import numpy as np
#import matplotlib.pyplot as plt
from random import randint
#import pandas as pd
import pickle

# load the model first and pass as an argument
model = VGG16()
model = Model(inputs = model.inputs, outputs = model.layers[-2].output)

def extract_features(imgarray, model):
    # load the image as a 224x224 array
    #img = load_img(file, target_size=(224,224))
    # convert from 'PIL.Image.Image' to numpy array
    #img = np.array(img) 
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    imgarray = imgarray.astype(np.uint8)
    img = array_to_img(imgarray)
    img = img.resize((224,224))
    img = np.array(img) 
    reshaped_img = img.reshape(1,224,224,3) 
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features

train_datagen=ImageDataGenerator()#rescale=1./255)
train_set=train_datagen.flow_from_directory('../classes/Superclass/hmbc/train',target_size=(1133,791),batch_size=8,color_mode='rgb',class_mode='categorical')

data = {}
classes = {}
i=0
for x,y in train_set:
	for z in range(8):
		#print(x[z].shape)
		#print(y[z].shape)
		feat = extract_features(x[z],model)
		data[i] = feat
		classes[i] = np.argmax(y[z])
		i+=1
		print(i)
		if i>=599:
			break
	else:
		continue
	break

feat = np.array(list(data.values()))
print(feat.shape)
# reshape so that there are 210 samples of 4096 vectors
feat = feat.reshape(-1,4096)
print(feat.shape)

# reduce the amount of dimensions in the feature vector
pca = PCA(n_components=100, random_state=22)
pca.fit(feat)
x = pca.transform(feat)

# cluster feature vectors
kmeans = KMeans(n_clusters=11, random_state=22)
kmeans.fit(x)

groups = {}
for file, cluster in zip(range(600),kmeans.labels_):
    if cluster not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(file)
    else:
        groups[cluster].append(file)

for i in groups[0]:
	print(classes[i])
