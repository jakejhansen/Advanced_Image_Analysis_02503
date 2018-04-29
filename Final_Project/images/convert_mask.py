import glob
import numpy as np
from shutil import copy2
import cv2


images = glob.glob('*.png')

print("Converting images images")
for i, img in enumerate(images):
    if i % 100 == 0:
        print("Processed: " + i)
        
    frame = cv2.imread(img)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 130, 33])
    upper_red = np.array([19, 255, 255])

    lower_red2 = np.array([164, 111, 0])
    upper_red2 = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_tot = mask | mask2

    mask_tot = cv2.erode(mask_tot, None, iterations=3)
    mask_tot = cv2.dilate(mask_tot, None, iterations=3)

    res = cv2.bitwise_and(frame, frame, mask=mask_tot)

    cv2.imwrite(img, res)

print("Done")