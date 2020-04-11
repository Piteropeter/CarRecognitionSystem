from keras.utils import plot_model
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import argparse
import random
import time

from src.Functions import *

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("-p", "--path", type=str, required=True, help="Path to database with images")
argument_parser.add_argument("-s", "--seed", type=int, default=2019, help="Seed for random generator")
argument_parser.add_argument("-i", "--image_size", type=int, default=56, help="Scales images to n*n pixels")
argument_parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size")
argument_parser.add_argument("-e", "--epochs", type=int, default=20, help="Epochs count")
argument_parser.add_argument("-a", "--augmentation",  action='store_const', const=1, help="Turn on augmentation")
args = vars(argument_parser.parse_args())

disable_tf_warnings()
print("[INFO] Initializing file scan")
file_scan_start = time.time()
files, brands = scan_files(args['path'])
print("[INFO] Scan complete")
print("[INFO]      Took " + str(round(time.time() - file_scan_start, 2)) + " s")

print("[INFO] Shuffling images")
image_paths = sorted(list(files))
random.seed(args['seed'])
random.shuffle(image_paths)

print("[INFO] Loading data")
file_load_start = time.time()
data, labels = load_files(image_paths, args['image_size'])
print("[INFO] Data loaded successfully!")
print("[INFO]      Took " + str(round(time.time() - file_load_start, 2)) + " s")

print("[INFO] Files loaded: ", len(data))
print("[INFO] Brand count: ", len(brands))

print("[INFO] Creating dataset")
dataset_create = time.time()
train_images, test_images, train_labels, test_labels = \
    train_test_split(data, labels, test_size=0.10, random_state=args['seed'])
print("[INFO] Dataset created!")
print("[INFO]      Took " + str(round(time.time() - dataset_create, 2)) + " s")

label_binarizer = LabelBinarizer()
train_labels = label_binarizer.fit_transform(train_labels)
test_labels = label_binarizer.transform(test_labels)

print("[INFO] Creating model")
network = create_model(args['image_size'], len(label_binarizer.classes_))

print("[INFO] Training network")
network_training_start = time.time()
results = train_network(network, train_images, test_images,
                        train_labels, test_labels, args)

print("[INFO] Training completed!")
print("[INFO]      Took " + str(round(time.time() - network_training_start, 2)) + " s")

print("[INFO] Evaluating network")
report = get_classification_report(network, label_binarizer, test_images, test_labels, args['batch_size'])
print(report)

print("[INFO] Saving network and results")

save_results(network, label_binarizer, results, report, args)

print("[INFO] Saving completed!")
print("[INFO] Closing program...")
