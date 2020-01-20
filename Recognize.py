import ast
import json
import pickle
import cv2
from keras.models import load_model
import tensorflow as tf
from zipfile import ZipFile

tf.get_logger().setLevel('FATAL')

zip_path = "./network.zip"
with ZipFile(zip_path, 'r') as zip_file:
    zip_file.extractall()

file = open("./info.json", "r")
data = file.read()
data = json.loads(data)
print(data)
IMAGE_SIZE = data['IMAGE_SIZE']

image = cv2.imread("VMMRdb/mini_austin_1975/1972 AUSTIN MINI COOPER_00C0C_3QPt1U5KBh5_600x450.jpg")
output = image.copy()
image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
image = image.astype("float") / 255.0
image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))

print("[INFO] loading network and label binarizer...")
# model = load_model("trained_models/network5m50b32s64augON")
model = load_model("./model")
# lb = pickle.loads(open("trained_models/label_binarizer5m50b32s64augON", "rb").read())
lb = pickle.loads(open("./label_binarizer", "rb").read())
# make a prediction on the image

preds = model.predict(image)

for i, pred in enumerate(preds[0]):
    text = "{}: {:.2f}%".format(lb.classes_[i], preds[0][i] * 100)
    print(text)

# find the class label index with the largest corresponding probability
# draw the class label + probability on the output image
best_recognition = preds.argmax(axis=1)[0]
text = "{}: {:.2f}%".format(lb.classes_[best_recognition], preds[0][best_recognition] * 100)
cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# show the output image
# cv2.imshow("Image", output)
# cv2.waitKey(0)

# file = open("./label_binarizer", "wb")
