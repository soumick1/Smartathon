import cv2
import numpy as np


def laplacian(img):
    kernel = np.ones((5, 5), np.uint8)
    lap_img = cv2.Laplacian(img, cv2.CV_32F)
    dilation = cv2.dilate(lap_img, kernel, iterations=1)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
    return closing
