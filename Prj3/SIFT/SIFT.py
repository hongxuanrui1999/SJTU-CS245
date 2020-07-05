import numpy as np
import pandas as pd
import cv2

IMG_PATH = '../AwA2-data/JPEGImages/'
LD_PATH = '../AwA2-data/SIFT_LD/'

def SIFT(className, imgName):
    image = cv2.imread(IMG_PATH + className + '/' + imgName + '.jpg')
    image = cv2.resize(image, (224, 224))
    sift = cv2.xfeatures2d_SIFT.create()
    kp, des = sift.detectAndCompute(image, None)
    return kp, des

def main():
    f_class_dict = np.load('../f_class_dict.npy', allow_pickle=True).item() #for load dict
    for className, totalNum in f_class_dict.items():
        print("SS at %s" % (className))
        for idx in range(10001, totalNum + 1):
            _, des = SIFT(className, className + '_' + str(idx))
            np.save(LD_PATH + className + '/' + className + '_' + str(idx), des)

if __name__ == '__main__':
    main()
