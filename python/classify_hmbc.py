from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import models
from keras import layers
from keras import backend as K
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
#import matplotlib.pyplot as plt

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(300, 205),color_mode='grayscale')
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    #if show:
    #    plt.imshow(img_tensor[0])                           
    #    plt.axis('off')
    #    plt.show()

    return img_tensor

test_datagen=ImageDataGenerator(rescale=1./255)
train_datagen=ImageDataGenerator(rescale=1./255)
test_set=test_datagen.flow_from_directory('../classesbothfinal/Superclass/hmbc/test',target_size=(1133,791),batch_size=105,color_mode='grayscale',class_mode='categorical')
train_set=train_datagen.flow_from_directory('../classesbothfinal/Superclass/hmbc/train',target_size=(1133,791),batch_size=8,color_mode='grayscale',class_mode='categorical')
#build network
network=models.Sequential()
network.add(layers.Conv2D(32, 3, activation='relu', input_shape=(1133,791, 1)))
network.add(layers.MaxPooling2D((2,2)))
network.add(layers.Conv2D(64, 3, activation='relu'))
network.add(layers.MaxPooling2D((2,2)))
network.add(layers.Conv2D(64, 3, activation='relu'))
network.add(layers.MaxPooling2D((2,2)))
network.add(layers.Conv2D(128, 3, activation='relu'))
network.add(layers.Flatten())
network.add(layers.Dense(64, activation='relu'))
network.add(layers.Dense(9, activation='softmax'))
#The default learning rate is 0.01
network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy',f1_m,precision_m, recall_m])
#perform training
network.fit(train_set, epochs=20)

x_test, y_test = next(test_set)
y_pred = network.predict(x_test, batch_size=105, verbose=1)
y_test_bool= np.argmax(y_test, axis=1)
y_pred_bool = np.argmax(y_pred, axis=1)
print( "matthews_corrcoef", matthews_corrcoef(y_test_bool, y_pred_bool)) 
#print(classification_report(y_test, y_pred))


score = network.evaluate(x_test, y_test) 
print('Test loss:', score[0]) 
print('Test accuracy:', score[1], score[2], score[3], score[4])
