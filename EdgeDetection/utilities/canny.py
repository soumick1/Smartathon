import cv2
import numpy as np


def canny(img):
    median = cv2.medianBlur(img, 5)
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(median, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
    edges = cv2.Canny(closing, 30, 80)
    return edges
