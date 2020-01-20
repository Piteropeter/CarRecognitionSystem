from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os
from zipfile import ZipFile
from keras import models
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D
import tensorflow as tf


def disable_tf_warnings():
    tf.get_logger().setLevel('FATAL')


def scan_files():
    files = []
    brands = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk("VMMRdb/"):
        for file in f:
            files.append(os.path.join(r, file))

        for directory in d:
            brand = directory.split('_')
            if brand[0] not in brands:
                brands.append(brand[0])
    return files, brands


def load_files(image_paths, IMAGE_SIZE):
    data = []
    labels = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        # image = cv2.normalize(image, norm_type=) TODO: ADD IMAGE NORMALIZATION (AND MAYBE HISTORGRAM EQUALIZATION)
        data.append(image)
        directory = image_path.split('/')[-1]
        label = directory.split('_')[0]
        labels.append(label)

    data = np.array(data, dtype="float") / 255.0
    return data, labels


def create_model(IMAGE_SIZE, label_count):
    network = models.Sequential()
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
    network.add(Dense(label_count))
    network.add(Activation("softmax"))

    network.compile(loss="categorical_crossentropy", optimizer=SGD(lr=0.01), metrics=["accuracy"])

    return network


def train_network(network, train_images, test_images, train_labels, test_labels, AUGMENTATION, EPOCHS, BATCH_SIZE):
    if AUGMENTATION:
        aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                                 height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                                 horizontal_flip=True, fill_mode="nearest")
        results = network.fit_generator(aug.flow(train_images, train_labels, batch_size=BATCH_SIZE),
                                        validation_data=(test_images, test_labels), steps_per_epoch=len(train_images),
                                        epochs=EPOCHS)
    else:
        results = network.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=EPOCHS,
                              batch_size=BATCH_SIZE)
    return results


def get_classification_report(network, label_binarizer, test_images, test_labels, BATCH_SIZE):
    predictions = network.predict(test_images, batch_size=BATCH_SIZE)
    return classification_report(test_labels.argmax(axis=1), predictions.argmax(axis=1),
                                 target_names=label_binarizer.classes_)


def save_results(network, label_binarizer, results, report, SEED, IMAGE_SIZE, BATCH_SIZE, EPOCHS, AUGMENTATION):
    network.save("./model")
    file = open("./label_binarizer", "wb")
    file.write(pickle.dumps(label_binarizer))
    file.close()

    file = open("classification_report.txt", "w")
    file.write(report)
    file.close()

    N = np.arange(0, EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, results.history["loss"], label="train_loss")
    plt.plot(N, results.history["val_loss"], label="val_loss")

    # To make it compatible with both tensorflow 1.x and 2.0
    try:
        plt.plot(N, results.history["accuracy"], label="train_acc")
        plt.plot(N, results.history["val_accuracy"], label="val_acc")
    except:
        plt.plot(N, results.history["acc"], label="train_acc")
        plt.plot(N, results.history["val_acc"], label="val_acc")

    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig("training_plot.png")

    file = open("./info.json", "w")
    info_json = "{\"SEED\" : " + str(SEED) + \
                ",\n\"IMAGE_SIZE\" : " + str(IMAGE_SIZE) + \
                ",\n\"BATCH_SIZE\" : " + str(BATCH_SIZE) + \
                ",\n\"EPOCHS\" : " + str(EPOCHS) + \
                ",\n\"AUGMENTATION\" : " + str(AUGMENTATION) + "}"
    file.write(info_json)
    file.close()

    zip_file = ZipFile('model.zip', 'w')
    zip_file.write('model')
    zip_file.write('label_binarizer')
    zip_file.write('info.json')
    zip_file.write('classification_report.txt')
    zip_file.write('training_plot.png')
    zip_file.close()
