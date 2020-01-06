from keras_preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import time
import cv2
import os
from keras import models
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D

SEED = 2019
IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 50
AUGMENTATION = 1

files = []
directories = []
brands = []

print("[INFO] Initializing file scan")
file_scan_start = time.time()

# r=root, d=directories, f = files
for r, d, f in os.walk("VMMRdb/"):
    for file in f:
        # if '.jpg' in file:
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
random.seed(SEED)
random.shuffle(image_paths)
print("[INFO] Loading data")
file_load_start = time.time()

for image_path in image_paths:
    image = cv2.imread(image_path)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    # image = cv2.normalize(image, norm_type=) TODO: ADD IMAGE NORMALIZATION (AND MAYBE HISTORGRAM EQUALIZATION)
    data.append(image)
    directory = image_path.split('/')[-1]
    label = directory.split('_')[0]
    labels.append(label)

print("[INFO] Data loaded successfully!")
print("[INFO]      Took " + str(round(time.time() - file_load_start, 2)) + " s")
print("[INFO] Files loaded: ", len(data))
print("[INFO] Brand count: ", len(brands))

# scale the raw pixel intensities to the range [0, 1]
print("[INFO] Creating dataset")
dataset_create = time.time()
data = np.array(data, dtype="float16") / 255.0
labels = np.array(labels)
(train_images, test_images, train_labels, test_labels) = train_test_split(data, labels, test_size=0.10, random_state=SEED)
print("[INFO] Dataset created!")
print("[INFO]      Took " + str(round(time.time() - dataset_create, 2)) + " s")

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

if AUGMENTATION:
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode="nearest")
    H = network.fit_generator(aug.flow(train_images, train_labels, batch_size=BATCH_SIZE),
                              validation_data=(test_images, test_labels), steps_per_epoch=len(train_images),
                              epochs=EPOCHS)
else:
    H = network.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=EPOCHS,
                    batch_size=BATCH_SIZE)

print("[INFO] Training completed!")
print("[INFO]      Took " + str(round(time.time() - network_training_start, 2)) + " s")
print("[INFO] Evaluating network")

predictions = network.predict(test_images, batch_size=BATCH_SIZE)
classification_report = classification_report(test_labels.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_)
print(classification_report)
file = open("classification_report.txt", "w")
file.write(classification_report)
file.close()

# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()

print(H.history)
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")

# To make it compatible with both tensorflow 1.x and 2.0
try:
    plt.plot(N, H.history["accuracy"], label="train_acc")
    plt.plot(N, H.history["val_accuracy"], label="val_acc")
except:
    plt.plot(N, H.history["acc"], label="train_acc")
    plt.plot(N, H.history["val_acc"], label="val_acc")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("training_plot.png")
print("[INFO] Saved results to training_plot.png and classification_report.txt")
print("[INFO] Serializing network and label binarizer")
network.save("./network")
file = open("./label_binarizer", "wb")
file.write(pickle.dumps(lb))
file.close()
print("[INFO] Serializing completed!")
