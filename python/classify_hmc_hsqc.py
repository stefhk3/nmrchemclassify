from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np

#This script needs the hmbc and hsqc spectrum from each compound. Either remove the hsqc spectra without hmbc, or add blank images for hmbc to make them match

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

train_datagen=ImageDataGenerator(rescale=1./255)
test_set_hmbc=train_datagen.flow_from_directory('../classesboth/Superclass/hmbc/test',target_size=(1133,791),batch_size=8,color_mode='grayscale',class_mode='categorical',shuffle=False)
test_set_hsqc=train_datagen.flow_from_directory('../classesboth/Superclass/hsqc/test',target_size=(1133,791),batch_size=8,color_mode='grayscale',class_mode='categorical',shuffle=False)

hmbc_imgs_test, hmbc_targets_test = test_set_hmbc.next()
hsqc_imgs_test, hsqc_targets_test = test_set_hsqc.next()

hmbc_input = keras.Input(
    shape=(1133,791, 1), name="hmbc"
) 
hsqc_input = keras.Input(
    shape=(1133,791, 1), name="hsqc"
)  
#the hmbc "column"
hmbc_conv1 = layers.Conv2D(32, (3,3), activation='relu', input_shape=(300, 205, 1))(hmbc_input)
hmbc_softmax1 = layers.MaxPooling2D(2,2)(hmbc_conv1)
hmbc_conv2 = layers.Conv2D(64, (3,3), activation='relu')(hmbc_softmax1)
hmbc_softmax2 = layers.MaxPooling2D(2,2)(hmbc_conv2)
hmbc_conv3 = layers.Conv2D(64, (3,3), activation='relu')(hmbc_softmax2)
hmbc_softmax3 = layers.MaxPooling2D(2,2)(hmbc_conv3)
hmbc_conv4 = layers.Conv2D(128, (3,3), activation='relu')(hmbc_softmax3)
hmbc_flatten = layers.Flatten()(hmbc_conv4)

#the hsqc "column"
hsqc_conv1 = layers.Conv2D(32, (3,3), activation='relu', input_shape=(300, 205, 1))(hsqc_input)
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
output = layers.Dense(17, activation='softmax', name='structure')(dense)

model = keras.Model(
    inputs=[hmbc_input, hsqc_input],
    outputs=[output],
)

#model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#I did not find a way to do the epochs automatically with all data, so I did it manually (note it is easy with feeding train_set directly, but we need to put together the two inputs)
epochs=10
for e in range(epochs):
    print('Epoch', e)
    train_set_hmbc=train_datagen.flow_from_directory('../classesboth/Superclass/hmbc/train',target_size=(1133,791),batch_size=32,color_mode='grayscale',class_mode='categorical',shuffle=False)
    train_set_hsqc=train_datagen.flow_from_directory('../classesboth/Superclass/hsqc/train',target_size=(1133,791),batch_size=32,color_mode='grayscale',class_mode='categorical',shuffle=False)
    hmbc_imgs, hmbc_targets = train_set_hmbc.next()
    hsqc_imgs, hsqc_targets = train_set_hsqc.next()
    model.fit(x=[hmbc_imgs, hsqc_imgs], y=hmbc_targets,  epochs=1, steps_per_epoch=11)

score = model.evaluate([hmbc_imgs_test,hsqc_imgs_test], hmbc_targets_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
