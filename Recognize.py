import cv2
import json
import pickle
import argparse
from keras.models import load_model
import tensorflow as tf
from zipfile import ZipFile

tf.get_logger().setLevel('FATAL')

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("-m", "--model", type=str, default="model.zip", help="Path to generated zip file")
argument_parser.add_argument("-i", "--image", type=str, required=True,
                             default="VMMRdb/mini_austin_1975/1972 AUSTIN MINI COOPER_00C0C_3QPt1U5KBh5_600x450.jpg",
                             help="Path to image to recognize")
args = vars(argument_parser.parse_args())

with ZipFile(args['model'], 'r') as zip_file:
    zip_file.extractall()

file = open("./info.json", "r")
data = file.read()
data = json.loads(data)
print(data)
image_size = data['IMAGE_SIZE']

image = cv2.imread(args['image'])
output = image.copy()
image = cv2.resize(image, (image_size, image_size))
image = image.astype("float") / 255.0
image = image.reshape((1, image_size, image_size, 3))

print("[INFO] Loading network and label binarizer...")
model = load_model("./model")
lb = pickle.loads(open("./label_binarizer", "rb").read())

predictions = model.predict(image)

for i, pred in enumerate(predictions[0]):
    text = "{}: {:.2f}%".format(lb.classes_[i], predictions[0][i] * 100)
    print(text)

# find the class label index with the largest corresponding probability
# draw the class label + probability on the output image
best_recognition = predictions.argmax(axis=1)[0]
text = "{}: {:.2f}%".format(lb.classes_[best_recognition], predictions[0][best_recognition] * 100)
cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# show the output image
# cv2.imshow("Image", output)
# cv2.waitKey(0)

# file = open("./label_binarizer", "wb")
