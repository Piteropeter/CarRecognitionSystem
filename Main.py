from keras_preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import time
import cv2
import os
from keras.datasets import mnist
from keras import models
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D
from keras import layers
from keras.utils import to_categorical
from opt_einsum import paths
import tensorflow as tf
import keras as keras

IMAGE_SIZE = 32
EPOCHS = 10

# config = tf.compat.v1.ConfigProto(device_count={'GPU': 1, 'CPU': 8})
# config.gpu_options.polling_inactive_delay_msecs = 100
# tf.compat.v1.global_variables.polling_inactive_delay_msecs = 100
# sess = tf.compat.v1.Session(config=config)
# keras.backend.set_session(sess)
# keras.backend.set_session()
# tf.compat.v1.keras.backend.set_session(sess)

files = []
directories = []
brands = []

print("[INFO] Initializing file scan")
file_scan_start = time.time()
# r=root, d=directories, f = files
for r, d, f in os.walk("VMMRdb/"):
    for file in f:
        if '.jpg' in file:
            files.append(os.path.join(r, file))

    for directory in d:
        directories.append(directory)
        brand = directory.split('_')
        if brand[0] not in brands:
            brands.append(brand[0])

print("[INFO] Scan complete")
print("[INFO]      Took " + str(round(time.time() - file_scan_start, 2)) + " s")
# exit(0)
data = []
labels = []
print("[INFO] Shuffling images")
image_paths = sorted(list(files))
random.seed(2019)
random.shuffle(image_paths)
print("[INFO] Loading data")
file_load_start = time.time()

# loop over the input images
for image_path in image_paths:
    # load the image, resize the image to be 32x32 pixels (ignoring
    # aspect ratio), flatten the image into 32x32x3=3072 pixel image
    # into a list, and store the image in the data list
    image = cv2.imread(image_path)
    # image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE)).flatten()
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    # print(len(image[0][0]))
    # if len(image[0][0]) > 3:
    #     exit(0)
    data.append(image)
    directory = image_path.split('/')[-1]
    # print(directory)
    label = directory.split('_')[0]
    # print(label)
    labels.append(label)

# for label in labels:
#     print(label)

print("[INFO] Data loaded successfully!")
print("[INFO]      Took " + str(round(time.time() - file_load_start, 2)) + " s")
print("[INFO] Files loaded: ", len(data))
print("[INFO] Brand count: ", len(brands))

# exit(0)

# for f in files:
#     print(f)

# for d in directories:
#     print(d)
#
# for b in brands:
#     print(b)

# scale the raw pixel intensities to the range [0, 1]
print("[INFO] Creating dataset")
dataset_create = time.time()
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
(train_images, test_images, train_labels, test_labels) = train_test_split(data, labels, test_size=0.30,
                                                                          random_state=2019)
print("[INFO] Dataset created!")
print("[INFO]      Took " + str(round(time.time() - dataset_create, 2)) + " s")


# class SmallVGGNet:
#     @staticmethod
#     def build(width, height, depth, classes):
# initialize the model along with the input shape to be
# "channels last" and the channels dimension itself


# if we are using "channels first", update the input shape
# and channels dimension
# if K.image_data_format() == "channels_first":
#     inputShape = (depth, height, width)
#     chanDim = 1

# convert the labels from integers to vectors (for 2-class, binary
# classification you should use Keras' to_categorical function
# instead as the scikit-learn's LabelBinarizer will not return a
# vector)
lb = LabelBinarizer()
train_labels = lb.fit_transform(train_labels)
test_labels = lb.transform(test_labels)

print("[INFO] Creating model")
network = models.Sequential()
# network.add(layers.Dense(1024, activation='relu', input_shape=(IMAGE_SIZE * IMAGE_SIZE * 3,)))
# network.add(layers.Dense(512, activation='relu', input_shape=(IMAGE_SIZE * IMAGE_SIZE * 3,)))
# network.add(layers.Dense(256, activation='relu', input_shape=(IMAGE_SIZE * IMAGE_SIZE * 3,)))
# network.add(layers.Dense(len(brands), activation='softmax'))


inputShape = (IMAGE_SIZE, IMAGE_SIZE, 3)
chanDim = -1
# CONV => RELU => POOL layer set
network.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
network.add(Activation("relu"))
network.add(BatchNormalization(axis=chanDim))
network.add(MaxPooling2D(pool_size=(2, 2)))
network.add(Dropout(0.25))
# (CONV => RELU) * 2 => POOL layer set
network.add(Conv2D(64, (3, 3), padding="same"))
network.add(Activation("relu"))
network.add(BatchNormalization(axis=chanDim))
network.add(Conv2D(64, (3, 3), padding="same"))
network.add(Activation("relu"))
network.add(BatchNormalization(axis=chanDim))
network.add(MaxPooling2D(pool_size=(2, 2)))
network.add(Dropout(0.25))
# (CONV => RELU) * 3 => POOL layer set
network.add(Conv2D(128, (3, 3), padding="same"))
network.add(Activation("relu"))
network.add(BatchNormalization(axis=chanDim))
network.add(Conv2D(128, (3, 3), padding="same"))
network.add(Activation("relu"))
network.add(BatchNormalization(axis=chanDim))
network.add(Conv2D(128, (3, 3), padding="same"))
network.add(Activation("relu"))
network.add(BatchNormalization(axis=chanDim))
network.add(MaxPooling2D(pool_size=(2, 2)))
network.add(Dropout(0.25))
# first (and only) set of FC => RELU layers
network.add(Flatten())
network.add(Dense(512))
network.add(Activation("relu"))
network.add(BatchNormalization())
network.add(Dropout(0.5))
# softmax classifier
network.add(Dense(len(lb.classes_)))
network.add(Activation("softmax"))

opt = SGD(lr=0.01)
# network.compile(optimizer='adam',
#                 loss='categorical_crossentropy',
#                 metrics=['accuracy'])
network.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
# test_labels = to_categorical(test_labels)
# train_labels = to_categorical(train_labels)
print("[INFO] Training network")
network_training_start = time.time()

# H = network.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=EPOCHS, batch_size=32)
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")
H = network.fit_generator(aug.flow(train_images, train_labels, batch_size=32),
                          validation_data=(test_images, test_labels), steps_per_epoch=len(train_images), epochs=EPOCHS)
print("[INFO] Training completed!")
print("[INFO]      Took " + str(round(time.time() - network_training_start, 2)) + " s")

# test_loss, test_acc = network.evaluate(test_images, test_labels)
# print('test_acc:', test_acc, 'test_loss', test_loss)

print("[INFO] Evaluating network")
predictions = network.predict(test_images, batch_size=32)
print(predictions)
print(len(predictions))
print("DUPA1")
print(test_labels)
print(len(test_labels))
print(classification_report(test_labels.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=lb.classes_))
print("DUPA2")
# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()

print(H.history)
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("./wykres")
