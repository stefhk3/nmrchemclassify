from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import models
from keras import layers
from keras import backend as K
from tensorflow import keras 
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

#This script needs the hmbc and hsqc spectrum from each compound. Either remove the hsqc spectra without hmbc, or add blank images for hmbc to make them match

def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(1133,791),color_mode='grayscale')
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    #if show:
    #    plt.imshow(img_tensor[0])                           
    #    plt.axis('off')
    #    plt.show()

    return img_tensor

class JoinedGenerator(keras.utils.Sequence):
    def __init__(self, generator1, generator2):
        self.generator1 = generator1
        self.generator2 = generator2 

    def __len__(self):
        return len(self.generator1)

    def __getitem__(self, i):
        x1, y1 = self.generator1[i]
        x2, y2 = self.generator2[i]
        return [x1, x2], y1

    def on_epoch_end(self):
        self.generator1.on_epoch_end()
        self.generator2.on_epoch_end()


train_datagen=ImageDataGenerator(rescale=1./255)
train_set_hmbc=train_datagen.flow_from_directory('../classesbothfinal/Superclass/hmbc/train',target_size=(1133,791),batch_size=8,color_mode='grayscale',class_mode='categorical',shuffle=False)
train_set_hsqc=train_datagen.flow_from_directory('../classesbothfinal/Superclass/hsqc/train',target_size=(1133,791),batch_size=8,color_mode='grayscale',class_mode='categorical',shuffle=False)
training_generator = JoinedGenerator(train_set_hmbc, train_set_hsqc)
test_set_hmbc=train_datagen.flow_from_directory('../classesbothfinal/Superclass/hmbc/test',target_size=(1133,791),batch_size=105,color_mode='grayscale',class_mode='categorical',shuffle=False)
test_set_hsqc=train_datagen.flow_from_directory('../classesbothfinal/Superclass/hsqc/test',target_size=(1133,791),batch_size=105,color_mode='grayscale',class_mode='categorical',shuffle=False)
testing_generator = JoinedGenerator(test_set_hmbc, test_set_hsqc)

hmbc_input = keras.Input(
    shape=(1133,791, 1), name="hmbc"
) 
hsqc_input = keras.Input(
    shape=(1133,791, 1), name="hsqc"
)  
#the hmbc "column"
hmbc_conv1 = layers.Conv2D(32, (3,3), activation='relu', input_shape=(1133,791, 1))(hmbc_input)
hmbc_softmax1 = layers.MaxPooling2D(2,2)(hmbc_conv1)
hmbc_conv2 = layers.Conv2D(64, (3,3), activation='relu')(hmbc_softmax1)
hmbc_softmax2 = layers.MaxPooling2D(2,2)(hmbc_conv2)
hmbc_conv3 = layers.Conv2D(64, (3,3), activation='relu')(hmbc_softmax2)
hmbc_softmax3 = layers.MaxPooling2D(2,2)(hmbc_conv3)
hmbc_conv4 = layers.Conv2D(128, (3,3), activation='relu')(hmbc_softmax3)
hmbc_flatten = layers.Flatten()(hmbc_conv4)

#the hsqc "column"
hsqc_conv1 = layers.Conv2D(32, (3,3), activation='relu', input_shape=(1133,791, 1))(hsqc_input)
hsqc_softmax1 = layers.MaxPooling2D(2,2)(hsqc_conv1)
hsqc_conv2 = layers.Conv2D(64, (3,3), activation='relu')(hsqc_softmax1)
hsqc_softmax2 = layers.MaxPooling2D(2,2)(hsqc_conv2)
hsqc_conv3 = layers.Conv2D(64, (3,3), activation='relu')(hsqc_softmax2)
hsqc_softmax3 = layers.MaxPooling2D(2,2)(hsqc_conv3)
hsqc_conv4 = layers.Conv2D(128, (3,3), activation='relu')(hsqc_softmax3)
hsqc_flatten = layers.Flatten()(hsqc_conv4)

#concetenate
concatted = layers.Concatenate()([hsqc_flatten, hmbc_flatten])

#the output
dense = layers.Dense(64, activation='relu')(concatted)
output = layers.Dense(9, activation='softmax', name='structure')(dense)

model = keras.Model(
    inputs=[hmbc_input, hsqc_input],
    outputs=[output],
)

#model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy',f1_m,precision_m, recall_m])

model.fit_generator(generator=training_generator,  epochs=30)

x_test, y_test = testing_generator[0]
y_pred = model.predict(x_test, batch_size=105, verbose=1)
y_test_bool= np.argmax(y_test, axis=1)
y_pred_bool = np.argmax(y_pred, axis=1)
print( "matthews_corrcoef", matthews_corrcoef(y_test_bool, y_pred_bool))
#print(classification_report(y_test, y_pred))


score = model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1], score[2], score[3], score[4])
