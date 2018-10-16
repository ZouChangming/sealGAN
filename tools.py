#encoding=utf-8

import os
import cv2
import numpy as np
import random

def get_classifier_pic(batch_size, seed):
    random.seed(seed)
    image = np.zeros(shape=(batch_size, 224, 224, 3), dtype=np.float32)
    label = np.zeros(shape=(batch_size, 2), dtype=np.float32)
    for i in range(batch_size):
        flag = random.choice([0, 1])
        if flag > 0:
            path = './data/source/seal'
            img_list = os.listdir(path)
            id = random.randint(0, len(img_list)-1)
            img = cv2.imread(os.path.join(path, img_list[id]))
            img = cv2.resize(img, (224, 224))
            image[i, :, :, :] = img
            label[i, 0] = 1
        else:
            path = './data/source/noseal'
            img_list = os.listdir(path)
            id = random.randint(0, len(img_list)-1)
            img = cv2.imread(os.path.join(path, img_list[id]))
            img = cv2.resize(img, (224, 224))
            image[i, :, :, :] = img
            label[i, 1] = 1
    return image, label
