#this implements https://colab.research.google.com/drive/1WiqyF7dCdnNBIANEY80Pxw_mVz4fyV-S?usp=sharing#scrollTo=tGSrH6W-PXMf

# imports
import os, sys
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageOps
import os.path

# third party imports
import numpy as np
import tensorflow as tf
assert tf.__version__.startswith('2.'), 'This tutorial assumes Tensorflow 2.0+'

# local imports
import voxelmorph as vxm
from plot import slices

imgs = []
path = "../classes/Superclass/hmbc/train/Benzenoids"
valid_images = [".jpg",".gif",".png",".tga"]
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    imgorig=ImageOps.grayscale(Image.open(os.path.join(path,f)))
    img=np.array(imgorig)
    pad_amount = ((175,170), (2,1))
    img = np.pad(img, pad_amount, 'constant')
    print(img.dtype)
    img=img.astype('float')/255
    imgs.append(img)

imgsnp = np.array(imgs)
print(imgsnp.shape)
print(imgsnp.dtype)

# configure unet input shape (concatenation of moving and fixed images)
ndim = 2
unet_input_features = 2
inshape = (*imgsnp.shape[1:], unet_input_features)
print(inshape)

# configure unet features 
nb_features = [
    [32, 32, 32, 32],         # encoder features
    [32, 32, 32, 32, 32, 16]  # decoder features
]

# build model
unet = vxm.networks.Unet(inshape=inshape, nb_features=nb_features)
print('input shape: ', unet.input.shape)
print('output shape:', unet.output.shape)

# transform the results into a flow field.
disp_tensor = tf.keras.layers.Conv2D(ndim, kernel_size=3, padding='same', name='disp')(unet.output)

# check tensor shape
print('displacement tensor:', disp_tensor.shape)

# using keras, we can easily form new models via tensor pointers
def_model = tf.keras.models.Model(unet.inputs, disp_tensor)

# build transformer layer
spatial_transformer = vxm.layers.SpatialTransformer(name='transformer')

# extract the first frame (i.e. the "moving" image) from unet input tensor
moving_image = tf.expand_dims(unet.input[..., 0], axis=-1)

# warp the moving image with the transformer
moved_image_tensor = spatial_transformer([moving_image, disp_tensor])

#is this needed?
outputs = [moved_image_tensor, disp_tensor]
vxm_model = tf.keras.models.Model(inputs=unet.inputs, outputs=outputs)

# build model using VxmDense
inshape = imgsnp.shape[1:]
vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=0)
print('input shape: ', ', '.join([str(t.shape) for t in vxm_model.inputs]))
print('output shape:', ', '.join([str(t.shape) for t in vxm_model.outputs]))

# voxelmorph has a variety of custom loss classes
losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]

# usually, we have to balance the two losses by a hyper-parameter
lambda_param = 0.05
loss_weights = [1, lambda_param]

vxm_model.compile(optimizer='Adam', loss=losses, loss_weights=loss_weights)

def vxm_data_generator(x_data, batch_size=32):
    """
    Generator that takes in data of size [N, H, W], and yields data for
    our custom vxm model. Note that we need to provide numpy data for each
    input, and each output.

    inputs:  moving [bs, H, W, 1], fixed image [bs, H, W, 1]
    outputs: moved image [bs, H, W, 1], zero-gradient [bs, H, W, 2]
    """

    # preliminary sizing
    vol_shape = x_data.shape[1:] # extract data shape
    ndims = len(vol_shape)
    
    # prepare a zero array the size of the deformation
    # we'll explain this below
    zero_phi = np.zeros([batch_size, *vol_shape, ndims],dtype=np.float32)
    print(zero_phi.dtype)
    while True:
        # prepare inputs:
        # images need to be of the size [batch_size, H, W, 1]
        idx1 = np.random.randint(0, x_data.shape[0], size=batch_size)
        moving_images = x_data[idx1, ..., np.newaxis]	
        idx2 = np.random.randint(0, x_data.shape[0], size=batch_size)
        fixed_images = x_data[idx2, ..., np.newaxis]
        inputs = [moving_images, fixed_images]
        # prepare outputs (the 'true' moved image):
        # of course, we don't have this, but we know we want to compare 
        # the resulting moved image with the fixed image. 
        # we also wish to penalize the deformation field. 
        outputs = [fixed_images, zero_phi]
        
        yield (inputs, outputs)

train_generator = vxm_data_generator(imgsnp,2)
nb_epochs = 20
steps_per_epoch = 100
hist = vxm_model.fit_generator(train_generator, epochs=nb_epochs, steps_per_epoch=steps_per_epoch, verbose=2);

#test with some benzenoids
val_generator = vxm_data_generator(imgsnp, batch_size = 1)
val_input, _ = next(val_generator)
val_pred = vxm_model.predict(val_input)
sum=0
for i in val_pred[1].squeeze():
  for k in i:
    for l in k:
      sum=sum+abs(l)
print("sum2 ",sum)
val_input, _ = next(val_generator)
val_pred = vxm_model.predict(val_input)
sum=0
for i in val_pred[1].squeeze():
  for k in i:
    for l in k:
      sum=sum+abs(l)
print("sum2 ",sum)
val_input, _ = next(val_generator)
val_pred = vxm_model.predict(val_input)
sum=0
for i in val_pred[1].squeeze():
  for k in i:
    for l in k:
      sum=sum+abs(l)
print("sum2 ",sum)

#and some other molecules
imgs = []
path = "../classes/Superclass/hmbc/train/Phenylpropanoids and polyketides"
valid_images = [".jpg",".gif",".png",".tga"]
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    imgorig=ImageOps.grayscale(Image.open(os.path.join(path,f)))
    img=np.array(imgorig)
    pad_amount = ((175,170), (2,1))
    img = np.pad(img, pad_amount, 'constant')
    img=img.astype('float')/255
    imgs.append(img)

imgsnp = np.array(imgs)
val_generator = vxm_data_generator(imgsnp, batch_size = 1)
val_input, _ = next(val_generator)
val_pred = vxm_model.predict(val_input)
sum=0
for i in val_pred[1].squeeze():
  for k in i:
    for l in k:
      sum=sum+abs(l)
print("sum2 ",sum)
val_input, _ = next(val_generator)
val_pred = vxm_model.predict(val_input)
sum=0
for i in val_pred[1].squeeze():
  for k in i:
    for l in k:
      sum=sum+abs(l)
print("sum2 ",sum)
val_input, _ = next(val_generator)
val_pred = vxm_model.predict(val_input)
sum=0
for i in val_pred[1].squeeze():
  for k in i:
    for l in k:
      sum=sum+abs(l)
print("sum2 ",sum)

