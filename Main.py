from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
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
from keras import layers
from keras.utils import to_categorical
from opt_einsum import paths

files = []
directories = []
brands = []

print("[INFO] Initializing file scan")
file_scan_start = time.time()
# r=root, d=directories, f = files
for r, d, f in os.walk("VMMRdb/"):
    for file in f:
        # if '.jpg' not in file:
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
    image = cv2.resize(image, (32, 32)).flatten()
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
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] Creating dataset")
dataset_create = time.time()
(train_images, test_images, train_labels, test_labels) = train_test_split(data, labels, test_size=0.30,
                                                                          random_state=2019)
print("[INFO] Dataset created!")
print("[INFO]      Took " + str(round(time.time() - dataset_create, 2)) + " s")

# convert the labels from integers to vectors (for 2-class, binary
# classification you should use Keras' to_categorical function
# instead as the scikit-learn's LabelBinarizer will not return a
# vector)
lb = LabelBinarizer()
train_labels = lb.fit_transform(train_labels)
test_labels = lb.transform(test_labels)
print("[INFO] Creating model")
network = models.Sequential()
network.add(layers.Dense(1024, activation='relu', input_shape=(32 * 32 * 3,)))
network.add(layers.Dense(512, activation='relu', input_shape=(32 * 32 * 3,)))
network.add(layers.Dense(256, activation='relu', input_shape=(32 * 32 * 3,)))
# network.add(layers.Dense(512, activation='sigmoid'))
network.add(layers.Dense(len(brands), activation='softmax'))
opt = SGD(lr=0.01)
network.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

print("[INFO] Training network")
network_training_start = time.time()
network.fit(train_images, train_labels, epochs=5, batch_size=32)
print("[INFO] Training completed!")
print("[INFO]      Took " + str(round(time.time() - network_training_start, 2)) + " s")

# test_loss, test_acc = network.evaluate(test_images, test_labels)
# print('test_acc:', test_acc, 'test_loss', test_loss)

print("[INFO] Evaluating network")
predictions = network.predict(test_images, batch_size=32)
print(classification_report(test_labels.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=lb.classes_))

# plot the training loss and accuracy
N = np.arange(0, 5)
plt.style.use("ggplot")
plt.figure()

H = network.fit(train_images, train_labels, validation_data=(test_images, test_labels),
                epochs=5, batch_size=32)

plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("./fig")
