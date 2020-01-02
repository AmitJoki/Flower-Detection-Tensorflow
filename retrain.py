from __future__ import absolute_import, division, print_function

import itertools
import os

import matplotlib.pylab as plt
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import pathlib

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

module_selection = ("mobilenet_v2", 224, 1280) #@param ["(\"mobilenet_v2\", 224, 1280)", "(\"inception_v3\", 299, 2048)"] {type:"raw", allow-input: true}
handle_base, pixels, FV_SIZE = module_selection
MODULE_HANDLE ="https://tfhub.dev/google/tf2-preview/{}/feature_vector/4".format(handle_base)
IMAGE_SIZE = (pixels, pixels)
print("Using {} with input size {} and output dimension {}".format(
  MODULE_HANDLE, IMAGE_SIZE, FV_SIZE))

BATCH_SIZE = 32 #@param {type:"integer"}

data_dir = 'Images/flower_photos'

datagen_kwargs = dict(rescale=1./255, validation_split=.20)
dataflow_kwargs = dict(target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
                   interpolation="bilinear")

valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    **datagen_kwargs)
valid_generator = valid_datagen.flow_from_directory(
    data_dir, subset="validation", shuffle=False, **dataflow_kwargs)

do_data_augmentation = True #@param {type:"boolean"}
if do_data_augmentation:
  train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
      rotation_range=40,
      horizontal_flip=True,
      width_shift_range=0.2, height_shift_range=0.2,
      shear_range=0.2, zoom_range=0.2,
      **datagen_kwargs)
else:
  train_datagen = valid_datagen
train_generator = train_datagen.flow_from_directory(
    data_dir, subset="training", shuffle=True, **dataflow_kwargs)


do_fine_tuning = False #@param {type:"boolean"}

def compile_model():
  model.compile(
    optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy'])

data_root = pathlib.Path(data_dir)
label_names = sorted([item.name for item in data_root.glob('*/') if item.is_dir()])

try:
  model = tf.keras.models.load_model('model.h5', custom_objects={'KerasLayer': hub.KerasLayer})
  compile_model()
except:
  print("Building model with", MODULE_HANDLE)
  model = tf.keras.Sequential([
      hub.KerasLayer(MODULE_HANDLE, output_shape=[FV_SIZE],
                    trainable=do_fine_tuning),
      tf.keras.layers.Dropout(rate=0.2),
      tf.keras.layers.Dense(train_generator.num_classes, activation='softmax',
                            kernel_regularizer=tf.keras.regularizers.l2(0.0001))
  ])
  model.build((None,)+IMAGE_SIZE+(3,))
  model.summary()
  compile_model()
  steps_per_epoch = train_generator.samples // train_generator.batch_size
  validation_steps = valid_generator.samples // valid_generator.batch_size
  hist = model.fit_generator(
      train_generator,
      epochs=10, steps_per_epoch=steps_per_epoch,
      validation_data=valid_generator,
      validation_steps=validation_steps).history

  tf.keras.models.save_model(model, 'model.h5', save_format='h5')

def predict(filename):
  image = tf.io.read_file(filename)
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [224, 224])
  image = tf.keras.preprocessing.image.img_to_array(image)
  image = tf.keras.applications.mobilenet_v2.preprocess_input(image.copy())
  image = tf.expand_dims(image, axis=0)
  print(enumerate(model.predict(image).tolist()[0]))
  return { map_index_to_label(i): x  for i, x in enumerate(model.predict(image).tolist()[0]) }

def map_index_to_label(idx):
  return label_names[idx]