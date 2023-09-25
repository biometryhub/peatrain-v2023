#!/usr/bin/python3
# New and streamlined version of peatrain -- Tensorflow Edition.
# Run in the working directory with
#   python peatrain_tensorflow.py
#
# Based on code developed by Lachlan Mitchell and Russell Edson,
# Biometry Hub, University of Adelaide.
# Date last modified: 21/07/2023

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf


# Neural Network model definition
image_height = 64
image_width = 64
input_shape = [image_height, image_width]
if tf.keras.backend.image_data_format() == "channels_first":
    input_shape.insert(0, 3)
else:
    input_shape.append(3)
input_shape = tuple(input_shape)

model = tf.keras.models.Sequential([
    tf.keras.layers.Rescaling(1.0 / 255, input_shape=input_shape),
    tf.keras.layers.Conv2D(
        filters=32, kernel_size=(5, 5), activation=tf.nn.relu
    ),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(
        filters=64, kernel_size=(5, 5), activation=tf.nn.relu
    ),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(
        filters=128, kernel_size=(5, 5), activation=tf.nn.relu
    ),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=1024, activation=tf.nn.relu),
    tf.keras.layers.Dropout(rate=0.3),
    tf.keras.layers.Dense(units=2, activation=tf.nn.softmax)
])
model.summary()

# Prepare training/validation data
image_directory = "images"
batch_size = 64
train_validation_split = [0.8, 0.2]

annotations = pd.read_csv("annotations.csv")
images = []
labels = []
for _, row in annotations.iterrows():
    image_filename = row["path"]
    image = cv2.imread(image_filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = tf.convert_to_tensor(image)
    images.append(image)
    labels.append(row["label"])

labels = tf.keras.utils.to_categorical(labels, num_classes=2)
images = tf.data.Dataset.from_tensor_slices((images, labels)).batch(batch_size)
training_data, validation_data = tf.keras.utils.split_dataset(
    images, train_validation_split[0], train_validation_split[1], shuffle=True
)

# Model training (+ computing accuracy/loss estimates per epoch)
epochs = 30
loss_fn = tf.keras.losses.CategoricalCrossentropy()
optimiser = tf.keras.optimizers.RMSprop(learning_rate=0.001)
acc_loss = pd.DataFrame({"epoch": [], "accuracy": [], "loss": []})

class RecordProgress(tf.keras.callbacks.Callback):
    def __init__(self, df_acc_loss):
        super(RecordProgress, self).__init__()
        self.acc_loss = df_acc_loss

    def on_epoch_end(self, epoch, logs):
        loss = logs["val_loss"]
        accuracy = logs["val_accuracy"]
        print(f"Validation:\nAccuracy:{accuracy:>7f}, loss:{loss:>7f}\n")
        self.acc_loss.loc[len(self.acc_loss)] = [epoch + 1, accuracy, loss]

model.compile(optimizer=optimiser, loss=loss_fn, metrics=["accuracy"])
model.fit(
    training_data,
    epochs=epochs,
    validation_data=validation_data,
    callbacks=[RecordProgress(acc_loss)]
)

# Save training accuracy/loss values
acc_loss_filename = "acc_loss.csv"
acc_loss.to_csv(acc_loss_filename, index=False)
print(f"Saved running accuracy/loss recording to {acc_loss_filename}")

# Save output model (as .keras for Tensorflow/Keras models)
model_filename = "nnmodel.keras"
model.save(model_filename)
print(f"Saved neural network model to {model_filename}")
