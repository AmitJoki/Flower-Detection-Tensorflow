from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import pathlib
import random

import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

data_root = pathlib.Path('Images/flower_photos')

all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

image_count = len(all_image_paths)

label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())

label_to_index = dict((name, index) for index,name in enumerate(label_names))

all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [192, 192])
  image = tf.image.random_brightness(image, 0.9)
  image /= 255.0  # normalize to [0,1] range
  return image

def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)

path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))

image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

BATCH_SIZE = 32

# Setting a shuffle buffer size as large as the dataset ensures that the data is
# completely shuffled.
ds = image_label_ds.cache(filename='./cache.tf-data')
ds = image_label_ds.shuffle(buffer_size=image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
# `prefetch` lets the dataset fetch batches, in the background while the model is training.
ds = ds.prefetch(buffer_size=AUTOTUNE)

steps_per_epoch=tf.math.ceil(len(all_image_paths)/BATCH_SIZE).numpy()
mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
mobile_net.trainable=False

def change_range(image,label):
  return 2*image-1, label

keras_ds = ds.map(change_range)
image_batch, label_batch = next(iter(keras_ds))
feature_map_batch = mobile_net(image_batch)

def train():
  model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])
  model.fit(ds, epochs=10, steps_per_epoch=1)
  model.save('model.h5')

try:
  model = tf.keras.models.load_model('model.h5')
except:
  model = tf.keras.Sequential([
    mobile_net,
    tf.keras.layers.GlobalMaxPooling2D(),
    tf.keras.layers.Dense(len(label_names))])

  train()

def predict():
  image = tf.io.read_file('target')
  image = preprocess_image(image)
  image = mobile_net(image)
  return model.predict_classes(image)[0]