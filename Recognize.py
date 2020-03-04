import cv2
import json
import pickle
import argparse
from zipfile import ZipFile
from keras.models import load_model
from src.Functions import disable_tf_warnings

disable_tf_warnings()

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("-m", "--model", type=str, required=True,
                             help="Path to generated zip file")
argument_parser.add_argument("-i", "--image", type=str, required=True,
                             help="Path to image to recognize")
argument_parser.add_argument("--minimal", action='store_const', const=1)
args = vars(argument_parser.parse_args())

with ZipFile(args['model'], 'r') as zip_file:
    zip_file.extractall()

file = open("./info.json", "r")
data = file.read()
data = json.loads(data)

image_size = data['IMAGE_SIZE']

image = cv2.imread(args['image'])
image = cv2.resize(image, (image_size, image_size))
image = cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX)
image = image.astype("float") / 255.0
image = image.reshape(1, image_size, image_size, 3)

model = load_model("./model")
lb = pickle.loads(open("./label_binarizer", "rb").read())

predictions = model.predict(image)

if args['minimal']:
    best_recognition = predictions.argmax(axis=1)[0]
    print(lb.classes_[best_recognition])
else:
    for i, pred in enumerate(predictions[0]):
        text = "{}: {:.2f}%".format(lb.classes_[i], predictions[0][i] * 100)
        print(text)
