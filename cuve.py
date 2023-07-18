import matplotlib.pyplot as plt
import matplotlib.image as mping
import numpy as np
import cv2
import os 
import glob
from moviepy.editor import VideoFileClip

image = cv2.imread("/Users/saksham/Desktop/SWACHAL/P1/test_images/IMG20230406175013.jpg")

def rgb_color_selection(iamge):
    lower_threshold = np.uint([200,200,200])
    upper_threshold = np.uint([255,255,255])
    white_mask = cv2.inRange(iamge,lower_threshold,upper_threshold)

    lower_threshold = np.uint([175,175,0])
    upper_threshold = np.uint([255,255,255])
    yellow_mask = cv2.inRange([image,lower_threshold,upper_threshold])

    mask = cv2.bitwise_or(white_mask,yellow_mask)
    masked_image = cv2.bitwise_and(image,image,mask=mask)

    return masked_image

image1 = rgb_color_selection(image)
cv2.imshow("ADAS",image1)
cv2.waitKey(0)