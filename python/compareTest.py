from IMGSimilarityMetrics import tf2MobileNetV2Compare
from IMGSimilarityMetrics import ORBCompare
from IMGSimilarityMetrics import SSICompare
# from IMGSimilarityMetrics import simpleITKCompare

def calculate(image1, image2):
   print("Image Similarity Using Machne Learning (MOBILENET V2)")
   distance = tf2MobileNetV2Compare(image1,image2,"CS")
   print(distance)
   distance = tf2MobileNetV2Compare(image1,image2,"E")
   print(distance)
   print("Image Similarity Using ORB/SIFT Feature Detector and Structural Similarity Index")
   distance = ORBCompare(image1,image2,50)
   print(distance)
   distance = SSICompare(image1,image2)
   print(distance)


print("compare alkaloids to themselves")
image1="../classes/Superclass/hmbc/test/Alkaloids and derivatives/(2D-HMBC)_bmse001010_nmr_set01_1H_13C_HMBC_ser.png"
image2="../classes/Superclass/hmbc/test/Alkaloids and derivatives/(2D-HMBC)_bmse001193_nmr_set01_8_ser.png"
image3="../classes/Superclass/hmbc/test/Alkaloids and derivatives/(2D-HMBC)_bmse001248_nmr_set01_8_ser.png"
image4="../classes/Superclass/hmbc/test/Alkaloids and derivatives/(2D-HMBC)_bmse001281_nmr_set01_8_ser.png"
calculate(image1,image1)
calculate(image1,image2)
calculate(image1,image3)
calculate(image1,image4)
print("compare Lipids and lipid-like molecules to themselves")
image5="../classes/Superclass/hmbc/test/Lipids and lipid-like molecules/(2D-HMBC)_bmse000317_nmr_set01_1H_13C_HMBC_ser_noZoom.png"
image6="../classes/Superclass/hmbc/test/Lipids and lipid-like molecules/(2D-HMBC)_bmse000394_nmr_set01_1H_13C_HMBC_ser_noZoom.png"
image7="../classes/Superclass/hmbc/test/Lipids and lipid-like molecules/(2D-HMBC)_bmse000478_nmr_set01_1H_13C_HMBC_ser_noZoom.png"
image8="../classes/Superclass/hmbc/test/Lipids and lipid-like molecules/(2D-HMBC)_bmse000484_nmr_set01_1H_13C_HMBC_ser_noZoom.png"
calculate(image5,image5)
calculate(image5,image6)
calculate(image5,image7)
calculate(image5,image8)
print("compare alkaloids to lipids")
calculate(image1,image5)
calculate(image1,image6)
calculate(image1,image7)
calculate(image1,image8)
calculate(image2,image5)
calculate(image2,image6)
calculate(image2,image7)
calculate(image2,image8)
calculate(image3,image5)
calculate(image3,image6)
calculate(image3,image7)
calculate(image3,image8)
calculate(image4,image5)
calculate(image4,image6)
calculate(image4,image7)
calculate(image4,image8)
print("compare Organic oxygen compounds to themselves")
image9="../classes/Superclass/hmbc/train/Organic oxygen compounds/(2D-HMBC)_bmse000302_nmr_set01_1H_13C_HMBC_ser_noZoom.png"
image10="../classes/Superclass/hmbc/train/Organic oxygen compounds/(2D-HMBC)_bmse000303_nmr_set01_1H_13C_HMBC_ser_noZoom.png"
image11="../classes/Superclass/hmbc/train/Organic oxygen compounds/(2D-HMBC)_bmse000304_nmr_set01_1H_13C_HMBC_ser_noZoom.png"
image12="../classes/Superclass/hmbc/train/Organic oxygen compounds/(2D-HMBC)_bmse000306_nmr_set01_1H_13C_HMBC_ser_noZoom.png"
calculate(image9,image9)
calculate(image9,image10)
calculate(image9,image11)
calculate(image9,image12)
print("compare alkaloids to oxygen comounts")
calculate(image1,image9)
calculate(image1,image10)
calculate(image1,image11)
calculate(image1,image12)
calculate(image2,image9)
calculate(image2,image10)
calculate(image2,image11)
calculate(image2,image12)
calculate(image3,image9)
calculate(image3,image10)
calculate(image3,image11)
calculate(image3,image12)
calculate(image4,image9)
calculate(image4,image10)
calculate(image4,image11)
calculate(image4,image12)
print("compare oxygen compounds to lipids")
calculate(image9,image5)
calculate(image9,image6)
calculate(image9,image7)
calculate(image9,image8)
calculate(image10,image5)
calculate(image10,image6)
calculate(image10,image7)
calculate(image10,image8)
calculate(image11,image5)
calculate(image11,image6)
calculate(image11,image7)
calculate(image11,image8)
calculate(image12,image5)
calculate(image12,image6)
calculate(image12,image7)
calculate(image12,image8)




#TO FIX
# print("Image Similarity Using Image Registration Methods from SimpleITK")
#
#
# distance = simpleITKCompare("../images/(2D-COSY)_bmse000001_nmr_set01_HH_COSY_ser.png","../images/(2D-COSY)_bmse000002_nmr_set01_HH_COSY_pH9.01_ser.png",'D')
# print(distance)
#
