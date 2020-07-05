import numpy as np
import matplotlib.pyplot as plt
import cv2

imageName = "../AwA2-data/JPEGImages/antelope/antelope_10001.jpg"
# imageName = "../ImgChanged/dolphin_10180.jpg"
# imageName = "../ImgChanged/humpback+whale_10428.jpg"

def drawDemo(imageName):
    image = cv2.imread(imageName)
    image = cv2.resize(image, (224, 224))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(image, None)
    result = cv2.drawKeypoints(gray_image, kp, None)
    plt.imshow(result)
    plt.savefig("Demo.jpg")

if __name__ == '__main__':
    drawDemo(imageName)
