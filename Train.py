from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import random
import time

from Functions import *

SEED = 2019
IMAGE_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 5
AUGMENTATION = 0

disable_tf_warnings()
print("[INFO] Initializing file scan")
file_scan_start = time.time()
files, brands = scan_files()
print("[INFO] Scan complete")
print("[INFO]      Took " + str(round(time.time() - file_scan_start, 2)) + " s")

print("[INFO] Shuffling images")
image_paths = sorted(list(files))
random.seed(SEED)
random.shuffle(image_paths)

print("[INFO] Loading data")
file_load_start = time.time()
data, labels = load_files(image_paths, IMAGE_SIZE)
print("[INFO] Data loaded successfully!")
print("[INFO]      Took " + str(round(time.time() - file_load_start, 2)) + " s")

print("[INFO] Files loaded: ", len(data))
print("[INFO] Brand count: ", len(brands))

print("[INFO] Creating dataset")
dataset_create = time.time()
train_images, test_images, train_labels, test_labels = train_test_split(data, labels, test_size=0.10, random_state=SEED)
print("[INFO] Dataset created!")
print("[INFO]      Took " + str(round(time.time() - dataset_create, 2)) + " s")

label_binarizer = LabelBinarizer()
train_labels = label_binarizer.fit_transform(train_labels)
test_labels = label_binarizer.transform(test_labels)

print("[INFO] Creating model")
# test_labels = to_categorical(test_labels)
# train_labels = to_categorical(train_labels)
network = create_model(IMAGE_SIZE, len(label_binarizer.classes_))
print("[INFO] Training network")
network_training_start = time.time()

results = train_network(network, train_images, test_images, train_labels, test_labels, AUGMENTATION, EPOCHS, BATCH_SIZE)

print("[INFO] Training completed!")
print("[INFO]      Took " + str(round(time.time() - network_training_start, 2)) + " s")
print("[INFO] Evaluating network")

report = get_classification_report(network, label_binarizer, test_images, test_labels, BATCH_SIZE)
print(report)

print("[INFO] Saving network and results")

save_results(network, label_binarizer, results, report, SEED, IMAGE_SIZE, BATCH_SIZE, EPOCHS, AUGMENTATION)

print("[INFO] Saving completed!")
print("[INFO] Closing program...")
