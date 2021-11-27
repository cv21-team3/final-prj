# Copyright (C) 2018-2019, BigVision LLC (LearnOpenCV.com), All Rights Reserved. 
# Author : Sunita Nayak
# Article : https://www.learnopencv.com/deep-learning-based-object-detection-and-instance-segmentation-using-mask-r-cnn-in-opencv-python-c/
# License: BSD-3-Clause-Attribution (Please read the license file.)
# This work is based on OpenCV samples code (https://opencv.org/license.html)    

import cv2 as cv
import argparse
import numpy as np
import os.path
import sys
import random
import os

# Initialize the parameters
confThreshold = 0.5  # Confidence threshold
maskThreshold = 0.1  # Mask threshold
classToDetect = ['person'] # Classes to detect

#classes = None
colors = []

# Extract the bounding box and mask for each detected object
def postprocess_segment(classes,colors, image, boxes, masks):
    # Output size of masks is NxCxHxW where
    # N - number of detected boxes
    # C - number of classes (excluding background)
    # HxW - segmentation shape
    numClasses = masks.shape[1]
    numDetections = boxes.shape[2]
    imageH = image.shape[0]
    imageW = image.shape[1]
    box_size=[]
    bbox_coord=[]
    for i in range(numDetections):
        box = boxes[0, 0, i]
        mask = masks[i]
        score = box[2]
        classId = int(box[1])
        if classes[classId] not in classToDetect:
            continue
        if score > confThreshold:
            # Extract the bounding box
            left = int(imageW * box[3])
            top = int(imageH * box[4])
            right = int(imageW * box[5])
            bottom = int(imageH * box[6])

            left = max(0, min(left, imageW - 1))
            top = max(0, min(top, imageH - 1))
            right = max(0, min(right, imageW - 1))
            bottom = max(0, min(bottom, imageH - 1))
            #bounding box 안겹치게
            #if len(bbox_coord)==0:
            #  classMask = mask[classId]
            #  box_size.append(((bottom-top)*(right-left))/(imageH*imageW))
            #  bbox_coord.append((left,top,right,bottom))
            #  drawBox(colors,image, i, left, top, right, bottom, classMask)
            #else:
            #  for j in range(len(bbox_coord)):
            #    if (bbox_coord[j][0]<=left) and (right<=bbox_coord[j][2]) and (bottom<=bbox_coord[j][3]) and (bbox_coord[j][1]<=top):
            #      pass
            # Extract the mask for the object
            #    else:
            classMask = mask[classId]
            box_size.append(((bottom-top)*(right-left))/(imageH*imageW))
            bbox_coord.append((left,top,right,bottom))
            # Draw bounding box, colorize and show the mask on the image
            drawBox(colors,image, i, left, top, right, bottom, classMask)

    box_size=np.array(box_size)
    return box_size,bbox_coord

def postprocess_black(image, boxes, masks, idx):
    # Output size of masks is NxCxHxW where
    # N - number of detected boxes
    # C - number of classes (excluding background)
    # HxW - segmentation shape
    numClasses = masks.shape[1]
    numDetections = boxes.shape[2]

    imageH = image.shape[0]
    imageW = image.shape[1]

    box = boxes[0, 0, idx]
    mask = masks[idx]
    score = box[2]
    classId = int(box[1])

    # Extract the bounding box
    left = int(imageW * box[3])
    top = int(imageH * box[4])
    right = int(imageW * box[5])
    bottom = int(imageH * box[6])

    left = max(0, min(left, imageW - 1))
    top = max(0, min(top, imageH - 1))
    right = max(0, min(right, imageW - 1))
    bottom = max(0, min(bottom, imageH - 1))

    # Extract the mask for the object
    classMask = mask[classId]

    # Draw bounding box, colorize and show the mask on the image
    image = leaveFigure(image, idx, left, top, right, bottom, classMask)

    return image

# Draw the predicted bounding box, colorize and show the mask on the image
def drawBox(colors,frame, idx, left, top, right, bottom, classMask):
    # Draw a bounding box
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

    cv.putText(frame, f'{str(idx)}', (left, top), cv.FONT_HERSHEY_SIMPLEX, 1.0, (1, 1, 1), 3) # cv2.putText(img, text, org, fontFace, fontScale, color, thickness)

    # Resize the mask, threshold, color and apply it on the image
    classMask = cv.resize(classMask, (right - left + 1, bottom - top + 1))
    mask = (classMask > maskThreshold)
    roi = frame[top:bottom + 1, left:right + 1][mask]

    # color = colors[classId%len(colors)]
    # Comment the above line and uncomment the two lines below to generate different instance colors
    colorIndex = random.randint(0, len(colors) - 1)
    color = colors[colorIndex]

    frame[top:bottom + 1, left:right + 1][mask] = ([0.3 * color[0], 0.3 * color[1], 0.3 * color[2]] + 0.7 * roi).astype(np.uint8)

    # Draw the contours on the image
    mask = mask.astype(np.uint8)
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(frame[top:bottom + 1, left:right + 1], contours, -1, color, 3, cv.LINE_8, hierarchy, 100)

# Paint the selected area
def leaveFigure(image, idx, left, top, right, bottom, classMask):
    # Resize the mask, threshold, color and apply it on the image
    classMask = cv.resize(classMask, (right - left + 1, bottom - top + 1))
    mask = classMask > maskThreshold

    # Draw the contours on the image
    mask = mask.astype(np.uint8)
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    diff = image.copy()
    cv.drawContours(diff[top:bottom + 1, left:right + 1], contours, -1, color=(0,0,0), thickness=cv.FILLED)
    image = cv.subtract(image, diff)

    return image

#if __name__ == '__main__':
#    main()
