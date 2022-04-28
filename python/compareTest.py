from IMGSimilarityMetrics import tf2MobileNetV2Compare
from IMGSimilarityMetrics import ORBCompare
from IMGSimilarityMetrics import SSICompare
# from IMGSimilarityMetrics import simpleITKCompare

print("Image Similarity Using Machne Learning (MOBILENET V2)")

distance = tf2MobileNetV2Compare("..\images\(2D-COSY)_bmse000001_nmr_set01_HH_COSY_ser.png","..\images\(2D-COSY)_bmse000002_nmr_set01_HH_COSY_pH9.01_ser.png","CS")
print(distance)

distance = tf2MobileNetV2Compare("..\images\(2D-COSY)_bmse000001_nmr_set01_HH_COSY_ser.png","..\images\(2D-COSY)_bmse000002_nmr_set01_HH_COSY_pH9.01_ser.png","E")
print(distance)



print("Image Similarity Using ORB/SIFT Feature Detector and Structural Similarity Index")


distance = ORBCompare('..\images\(2D-COSY)_bmse000001_nmr_set01_HH_COSY_ser.png','..\images\(2D-COSY)_bmse000002_nmr_set01_HH_COSY_pH9.01_ser.png',50)
print(distance)

distance = SSICompare('..\images\(2D-COSY)_bmse000001_nmr_set01_HH_COSY_ser.png','..\images\(2D-COSY)_bmse000002_nmr_set01_HH_COSY_pH9.01_ser.png')
print(distance)




#TO FIX
# print("Image Similarity Using Image Registration Methods from SimpleITK")
#
#
# distance = simpleITKCompare("..\images\(2D-COSY)_bmse000001_nmr_set01_HH_COSY_ser.png","..\images\(2D-COSY)_bmse000002_nmr_set01_HH_COSY_pH9.01_ser.png",'D')
# print(distance)
#
