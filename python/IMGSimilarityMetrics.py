import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os

from scipy import spatial

global embed
embed = hub.KerasLayer(os.getcwd())

class TensorVector(object):

    def __init__(self, FileName=None):
        self.FileName = FileName

    def process(self) -> object:

        img = tf.io.read_file(self.FileName)
        img = tf.io.decode_png(img, channels=0)
        # 0: Use the number of channels in the PNG - encoded image.
        # 1: output a grayscale image.
        # 3: output an RGB image.
        # 4: output an RGBA image.
        img = tf.image.resize_with_pad(img, 224, 224)
        img = tf.image.convert_image_dtype(img,tf.float32)[tf.newaxis, ...]
        features = embed(img)
        feature_set = np.squeeze(features)
        return list(feature_set)





def tf2MobileNetV2Compare(img1,img2, metric):
    v1 = TensorVector(img1).process()
    v2 = TensorVector(img2).process()
    if metric == "CS":#Cosine Similarity
            distance =  1 - spatial.distance.cosine(v1, v2)
    elif metric == "E":
        distance = spatial.distance.euclidean(v1, v2)
    return distance #PEARSON CORRELATION?








import cv2

def ORBCompare(img1,img2,acceptbleSimilarity):
    image1 = cv2.imread(img1,0)
    image2 = cv2.imread(img2, 0)
    orb = cv2.ORB_create()
    kp1, desc1 = orb.detectAndCompute(image1, None)
    kp1 = np.asarray(kp1)
    kp2, desc2 = orb.detectAndCompute(image2, None)
    kp2 = np.asarray(kp2)
    bruteForce = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
    matches = bruteForce.match(desc1, desc2)

    similarAreas = [i for i in matches if i.distance < int(acceptbleSimilarity)]
    if len(matches) == 0:
        return 0

    return len(similarAreas) / len(matches)






from skimage.metrics import structural_similarity

def SSICompare(img1,img2):
    image1 = cv2.imread(img1,0)
    image2 = cv2.imread(img2, 0)
    similarity, diff = structural_similarity(image1, image2, full=True)
    return similarity






import SimpleITK as sitk


def simpleITKCompare(img1,img2, metric):
    image1 = sitk.ReadImage(img1, sitk.sitkFloat64)
    image2 = sitk.ReadImage(img2, sitk.sitkFloat64)
    registrationMethod = sitk.ImageRegistrationMethod()
    registrationMethod.Se
    if metric == "MI":#Mutual Information
        registrationMethod.SetMetricAsMattesMutualInformation()
        distance =  registrationMethod.MetricEvaluate(image1,image2)
    elif metric == "MS":#Mean Squares
        registrationMethod.SetMetricAsMeanSquares()
        distance = registrationMethod.MetricEvaluate(image1, image2)
    elif metric == "D":  # Demons
        registrationMethod.SetMetricAsDemons()
        distance = registrationMethod.MetricEvaluate(image1, image2)
    elif metric == "C":  # Correlation
        registrationMethod.SetMetricAsCorrelation()
        distance = registrationMethod.MetricEvaluate(image1, image2)
    elif metric == "NC":  # Neighbourhood Correlation
        registrationMethod.SetMetricAsANTSNeighborhoodCorrelation()
        distance = registrationMethod.MetricEvaluate(image1, image2)
    elif metric == "JHMI":  # Joint Histogram Mutual Information
        registrationMethod.SetMetricAsJointHistogramMutualInformation()
        distance = registrationMethod.MetricEvaluate(image1, image2)


