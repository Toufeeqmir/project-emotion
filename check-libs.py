import tensorflow as tf
import numpy as np
import matplotlib
import PIL
import cv2

print("TensorFlow", tf.__version__)
print("NumbPy:", np.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Pillow:",PIL.__version__)
print("OpenCV:",cv2.__version__)



import os

dataset_path = r"E:/python/dataset/train"
for emotion in os.listdir(dataset_path):
    path = os.path.join(dataset_path, emotion)
    print(emotion, "→", len(os.listdir(path)), "images")
