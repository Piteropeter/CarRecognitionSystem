from imutils import paths
import numpy as np
import argparse
import random
import pickle
import time
import cv2
import os

files = []
directories = []
brands = {}

print("[INFO] Initializing file scan")
file_scan_start = time.time()
# r=root, d=directories, f = files
for r, d, f in os.walk("VMMRdb/"):
    for directory in d:
        directories.append(directory)
        brand = directory.split('_')
        if brand[0] not in brands:
            brands[brand[0]] = 0

    for file in f:
        # if '.jpg' not in file:
        directory = os.path.join(r, file).split('/')[-1]
        # print(directory)
        label = str(directory.split('_')[0])
        brands[label] += 1
        files.append(os.path.join(r, file))

print("[INFO] Scan complete")
print("[INFO]      Took " + str(round(time.time() - file_scan_start, 2)) + " s")

print("Liczba marek: " + str(len(brands)))
for brand in brands:
    print(brand + ": " + str(brands[brand]))
