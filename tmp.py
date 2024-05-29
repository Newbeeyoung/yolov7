import pickle
import cv2
import numpy as np
import pdb

noise_path="runs/train/yolov7-em07123/EM-def-noise.pkl"
with open(noise_path, 'rb') as f:
    raw_noise = pickle.load(f)

pdb.set_trace()
cv2.imwrite("sample0.jpg",raw_noise[0])
