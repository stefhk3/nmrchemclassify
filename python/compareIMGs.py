import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt
import base64
from PIL import Image
import io
import math
from math import sqrt


global embed
embed = hub.KerasLayer(os.getcwd())

for x in os.listdir("."):
    print(x)

class TensorVector(object):

    def __init__(self, FileName=None):
        self.FileName = FileName

    def process(self):

        img = tf.io.read_file(self.FileName)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize_with_pad(img, 224, 224)
        img = tf.image.convert_image_dtype(img,tf.float32)[tf.newaxis, ...]
        features = embed(img)
        feature_set = np.squeeze(features)
        return list(feature_set)
#
#
def convertBase64(FileName):
    """
    Return the Numpy array for a image
    """
    with open(FileName, "rb") as f:
        data = f.read()

    res = base64.b64encode(data)

    base64data = res.decode("UTF-8")

    imgdata = base64.b64decode(base64data)

    image = Image.open(io.BytesIO(imgdata))

    return np.array(image)



helper = TensorVector("..\images\(2D-COSY)_bmse000001_nmr_set01_HH_COSY_ser.png")
vector = helper.process()

print(len(vector))

plt.imshow(convertBase64("..\images\(2D-COSY)_bmse000001_nmr_set01_HH_COSY_ser.png"))


helper = TensorVector("..\images\(2D-COSY)_bmse000002_nmr_set01_HH_COSY_pH9.01_ser.png")
vector2 = helper.process()

print(len(vector2))


def cosineSim(a1,a2):
    sum = 0
    suma1 = 0
    sumb1 = 0
    for i,j in zip(a1, a2):
        suma1 += i * i
        sumb1 += j*j
        sum += i*j
    cosine_sim = sum / ((sqrt(suma1))*(sqrt(sumb1)))
    return cosine_sim


cosineSim(vector, vector2)
print(cosineSim(vector, vector2))
# print(cosine_similarity(vector, vector2))