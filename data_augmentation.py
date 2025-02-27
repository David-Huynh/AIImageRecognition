import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import keras

import tensorflow as tf
from tensorflow import data as tf_data
from tensorflow import image as tf_image

from preprocess_common import *

PATH="archive/"

#Hyperparameters
AUTO = tf_data.AUTOTUNE
BATCH_SIZE = 16
EPOCHS = 10
VALIDATION_SIZE = 15000

class_names=['Human', 'AI']

train_df = pd.read_csv('./archive/train.csv', index_col=0)
test_df = pd.read_csv('./archive/test.csv')

train_paths = train_df["file_name"].array
train_labels = train_df["label"].array
test_paths = test_df["id"].array


#Split into training and validation
training_validation_paths = train_paths[:VALIDATION_SIZE]
training_validation_labels = keras.ops.one_hot(train_labels[:VALIDATION_SIZE],2)
training_paths = train_paths[VALIDATION_SIZE:]
training_labels = keras.ops.one_hot(train_labels[VALIDATION_SIZE:],2)

# Shuffles and batches the datasets
train_ds_one = (
    tf.data.Dataset.from_tensor_slices((training_paths, training_labels))
    .shuffle(1024)
    .map(lambda filename, label: resize_flip_scale_image(PATH+filename, label), num_parallel_calls=AUTO)
)
train_ds_two = (
    tf.data.Dataset.from_tensor_slices((training_paths, training_labels))
    .shuffle(1024)
    .map(lambda filename, label: resize_flip_scale_image(PATH+filename, label), num_parallel_calls=AUTO)
)

# Combine the two datasets
train_ds = tf_data.Dataset.zip((train_ds_one, train_ds_two))

val_ds = (
    tf_data.Dataset.from_tensor_slices((training_validation_paths, training_validation_labels))
    .map(lambda filename, label: resize_flip_scale_image(PATH+filename, label), num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

# Test dataset is unlabelled so 
# test_ds = (
#     tf_data.Dataset.from_tensor_slices((test_paths, test_labels))
#     .map(lambda filename, label: resize_flip_scale_image(PATH+filename, label), num_parallel_calls=AUTO)
#     .batch(BATCH_SIZE)
# )

mixer = Mix(img_size=224)
train_ds_cmu = (
    train_ds.shuffle(1024)
    .map(mixer.cutmix, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

