#this implements https://keras.io/examples/vision/semantic_image_clustering/

from collections import defaultdict
import random
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import itertools
#import matplotlib.pyplot as plt
from tqdm import tqdm

num_classes = 17
input_shape = (1133,791, 3)

test_datagen=ImageDataGenerator(rescale=1./255)
train_datagen=ImageDataGenerator(rescale=1./255)
test_set=test_datagen.flow_from_directory('../classes/Superclass/hmbc/test',target_size=(1133,791),batch_size=8,color_mode='rgb',class_mode='categorical')
train_set=train_datagen.flow_from_directory('../classes/Superclass/hmbc/train',target_size=(1133,791),batch_size=8,color_mode='rgb',class_mode='categorical')
#x_test, y_test = next(test_set)
#x_train, y_train = next(train_set)
#x_data = np.concatenate([x_train, x_test])
#y_data = np.concatenate([y_train, y_test])

#print("x_data shape:", x_data.shape, "- y_data shape:", y_data.shape)

classes = list(train_set.class_indices)

print(classes)

target_size = (1133,791)#32  # Resize the input images.
representation_dim = 512  # The dimensions of the features vector.
projection_units = 128  # The projection head of the representation learner.
num_clusters = 20  # Number of clusters.
k_neighbours = 5  # Number of neighbours to consider during cluster learning.
tune_encoder_during_clustering = False  # Freeze the encoder in the cluster learning.

data_preprocessing = keras.Sequential(
    [
        layers.Resizing(*target_size),
        layers.Normalization(),
    ]
)
# Compute the mean and the variance from the data for normalization.
#data_preprocessing.layers[-1].adapt(x_data)

#data_augmentation = keras.Sequential(
#    [
#        layers.RandomTranslation(
#            height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2), fill_mode="nearest"
#        ),
#        layers.RandomFlip(mode="horizontal"),
#        layers.RandomRotation(
#            factor=0.15, fill_mode="nearest"
#        ),
#        layers.RandomZoom(
#            height_factor=(-0.3, 0.1), width_factor=(-0.3, 0.1), fill_mode="nearest"
#        )
#    ]
#)

def create_encoder(representation_dim):
    encoder = keras.Sequential(
        [
            keras.applications.ResNet50V2(
                include_top=False, weights=None, pooling="avg"
            ),
            layers.Dense(representation_dim),
        ]
    )
    return encoder

class RepresentationLearner(keras.Model):
    def __init__(
        self,
        encoder,
        projection_units,
        num_augmentations,
        temperature=1.0,
        dropout_rate=0.1,
        l2_normalize=False,
        **kwargs
    ):
        super(RepresentationLearner, self).__init__(**kwargs)
        self.encoder = encoder
        # Create projection head.
        self.projector = keras.Sequential(
            [
                layers.Dropout(dropout_rate),
                layers.Dense(units=projection_units, use_bias=False),
                layers.BatchNormalization(),
                layers.ReLU(),
            ]
        )
        self.num_augmentations = num_augmentations
        self.temperature = temperature
        self.l2_normalize = l2_normalize
        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def compute_contrastive_loss(self, feature_vectors, batch_size):
        num_augmentations = tf.shape(feature_vectors)[0] // batch_size
        if self.l2_normalize:
            feature_vectors = tf.math.l2_normalize(feature_vectors, -1)
        # The logits shape is [num_augmentations * batch_size, num_augmentations * batch_size].
        logits = (
            tf.linalg.matmul(feature_vectors, feature_vectors, transpose_b=True)
            / self.temperature
        )
        # Apply log-max trick for numerical stability.
        logits_max = tf.math.reduce_max(logits, axis=1)
        logits = logits - logits_max
        # The shape of targets is [num_augmentations * batch_size, num_augmentations * batch_size].
        # targets is a matrix consits of num_augmentations submatrices of shape [batch_size * batch_size].
        # Each [batch_size * batch_size] submatrix is an identity matrix (diagonal entries are ones).
        targets = tf.tile(tf.eye(batch_size), [num_augmentations, num_augmentations])
        # Compute cross entropy loss
        return keras.losses.categorical_crossentropy(
            y_true=targets, y_pred=logits, from_logits=True
        )

    def call(self, inputs):
        # Preprocess the input images.
        preprocessed = data_preprocessing(inputs)
        # Create augmented versions of the images.
        #augmented = []
        #for _ in range(self.num_augmentations):
        #    augmented.append(data_augmentation(preprocessed))
        #augmented = layers.Concatenate(axis=0)(augmented)
        # Generate embedding representations of the images.
        features = self.encoder(inputs)
        # Apply projection head.
        return self.projector(features)

    def train_step(self, inputs):
        inputs=inputs[0]
        batch_size = tf.shape(inputs)[0]
        # Run the forward pass and compute the contrastive loss
        with tf.GradientTape() as tape:
            feature_vectors = self(inputs, training=True)
            loss = self.compute_contrastive_loss(feature_vectors, batch_size)
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update loss tracker metric
        self.loss_tracker.update_state(loss)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, inputs):
        inputs=inputs[0]
        batch_size = tf.shape(inputs)[0]
        feature_vectors = self(inputs, training=False)
        loss = self.compute_contrastive_loss(feature_vectors, batch_size)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

# Create vision encoder.
encoder = create_encoder(representation_dim)
# Create representation learner.
representation_learner = RepresentationLearner(
    encoder, projection_units, num_augmentations=2, temperature=0.1
)
# Create a a Cosine decay learning rate scheduler.
lr_scheduler = keras.optimizers.schedules.ExponentialDecay (initial_learning_rate=0.001,
 decay_steps=100000,
    decay_rate=0.96,
    staircase=True)
#    initial_learning_rate=0.001, decay_steps=500, alpha=0.1
#)
# Compile the model.
representation_learner.compile(
    optimizer=tfa.optimizers.AdamW(learning_rate=lr_scheduler, weight_decay=0.0001),
)
# Fit the model.
history = representation_learner.fit(train_set, epochs=1, validation_data=test_set)
#    x=x_data,
#    batch_size=512,
#    epochs=5,  # for better results, increase the number of epochs to 500.
#)

batch_size = 500
# Get the feature vector representations of the images.
feature_vectors = encoder.predict(train_set, verbose=1)
# Normalize the feature vectores.
feature_vectors = tf.math.l2_normalize(feature_vectors, -1)

neighbours = []
num_batches = feature_vectors.shape[0] // batch_size
for batch_idx in tqdm(range(num_batches)):
    start_idx = batch_idx * batch_size
    end_idx = start_idx + batch_size
    current_batch = feature_vectors[start_idx:end_idx]
    # Compute the dot similarity.
    similarities = tf.linalg.matmul(current_batch, feature_vectors, transpose_b=True)
    # Get the indices of most similar vectors.
    _, indices = tf.math.top_k(similarities, k=k_neighbours + 1, sorted=True)
    # Add the indices to the neighbours.
    neighbours.append(indices[..., 1:])

neighbours = np.reshape(np.array(neighbours), (-1, k_neighbours))
print(neighbours)
nrows = 4
ncols = k_neighbours + 1

classes=[]
for x,y in train_set:
    #print(y,"\n")
    for onehot in y:
       	classes.append(np.argmax(onehot))
        #print(onehot)
        #print(np.argmax(onehot))
        print(len(classes))
        if len(classes)>1000:
             break
    else:
        continue  # only executed if the inner loop did NOT break
    break  # only executed if the inner loop DID break

for i in neighbours:
    for j in i:
        print(j," ", classes[j],", ")
    print("\n")


