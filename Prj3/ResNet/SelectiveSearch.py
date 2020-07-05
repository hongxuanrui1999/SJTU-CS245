import numpy as np
import pickle
import time
from skimage import io, transform
import selective_search
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2

IMG_PATH = '../AwA2-data/JPEGImages/'

def gray2rgb():
    src = cv2.imread("../collie_10718.jpg", cv2.IMREAD_COLOR)
    cv2.imwrite("../collie_10718_rgb.jpg", src)

def SelectiveSearchImg(className, imgName):
    image = io.imread(IMG_PATH + className + '/' + imgName + '.jpg')

    image = transform.resize(image, (224, 224))
    boxes = selective_search.selective_search(image, mode='single')
    boxes_filter = selective_search.box_filter(boxes, min_size=30, topN=20)
    image = np.asarray(image)
    proposals = []
    for box in boxes_filter:
        w, h = box[2] - box[0], box[3] - box[1]
        if w < 250 and h < 250:
            img = cv2.resize(image[box[0] : box[2], box[1] : box[3], :], (224, 224))
            temp = np.empty((224, 224, 3))
            temp[:, :, 0] = img[:, :, 0]
            temp[:, :, 1] = img[:, :, 1]
            temp[:, :, 2] = img[:, :, 2]
            img = temp
            proposals.append(img)
    return proposals


