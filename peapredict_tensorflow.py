#!/usr/bin/python3
# Given an NN model and image(s), predict a plant. Tensorflow Edition.
# Run with
#   python peapredict_tensorflow.py MODEL IMAGE...
#
# Based on code developed by Lachlan Mitchell and Russell Edson,
# Biometry Hub, University of Adelaide.
# Date last modified: 16/07/2023

import sys
import cv2
import tensorflow as tf


model_filename = sys.argv[1]
image_filenames = sys.argv[2:]

# Load model
model = tf.keras.models.load_model(model_filename)

# Read in image(s) and predict
for image_filename in image_filenames:
    image = cv2.imread(image_filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.expand_dims(image, axis=0)

    pred = model.predict(image, steps=1)
    not_plant = pred[0][0]
    plant = pred[0][1]
    print(f"{image_filename}  Not plant: {not_plant:>7f}, Plant: {plant:>7f}")
