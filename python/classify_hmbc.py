from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import models
from keras import layers
import numpy as np
#import matplotlib.pyplot as plt

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
test_set=test_datagen.flow_from_directory('test',target_size=(1024,1024),batch_size=8,class_mode='binary')
train_set=train_datagen.flow_from_directory('../classes/Superclass/hmbc',target_size=(1133,791),batch_size=8,color_mode='grayscale',class_mode='categorical')
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
network.add(layers.Dense(17, activation='softmax'))
#The default learning rate is 0.01
network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#perform training
network.fit(train_set, epochs=50)

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
test_generator.reset()
pred=model.predict_generator(test_generator,steps=STEP_SIZE_TEST,verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)
labels=(validation_generator.class_indices)
labels2=dict((v,k) for k,v in labels.items())
predictions=[labels2[k] for k in predicted_class_indices]
print(predicted_class_indices)
print(labels)
print(predictions)
