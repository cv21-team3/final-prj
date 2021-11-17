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

classes = None
colors = []

def main():
    global classes, colors

    parser = argparse.ArgumentParser(description='Use this script to run Mask-RCNN object detection and segmentation')
    parser.add_argument('--image', help='Path to image file')
    parser.add_argument('--mode')
    args = parser.parse_args()

    # ==================== Load names of classes  ====================
    classesFile = "mscoco_labels.names"
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    # ==================== Load the network ====================
    textGraph = "./mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
    modelWeights = "./mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb"

    net = cv.dnn.readNetFromTensorflow(modelWeights, textGraph)
    net.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)

    #  ==================== Load the colors ====================
    colorsFile = "colors.txt"
    with open(colorsFile, 'rt') as f:
        colorsStr = f.read().rstrip('\n').split('\n')
    for i in range(len(colorsStr)):
        rgb = colorsStr[i].split(' ')
        color = np.array([float(rgb[0]), float(rgb[1]), float(rgb[2])])
        colors.append(color)

    winName = 'Object detection'
    cv.namedWindow(winName, cv.WINDOW_NORMAL)

    #  ==================== Open the image file  ====================
    if not os.path.isfile(args.image):
        print('Input image file ', args.image, ' does not exist')
        sys.exit(1)
    img = cv.imread(args.image)

    # ==================== Process the image file  ====================
    blob = cv.dnn.blobFromImage(img, swapRB=True, crop=False)  # Create a 4D blob from a frame
    net.setInput(blob)  # Set the input to the network
    boxes, masks = net.forward(['detection_out_final', 'detection_masks'])  # Run the forward pass to get output from the output layers

    original_img = img.copy()
    postprocess_segment(img, boxes, masks)  # Extract the bounding box and mask for each of the detected objects

    # ==================== Print the result  ====================
    if args.mode == 'segment':
        fileName = args.image[:-4] + '_segmented.jpg'
        cv.imwrite(fileName, img.astype(np.uint8))
    # ==================== Leave a target person only  ====================
    if args.mode == 'black':
        cv.imshow(winName, img.astype(np.uint8))
        cv.resizeWindow(winName, 1000, 800)
        cv.waitKey(1)
        preservedFigures = []
        idx = int(input('Enter the index of the figure to preserve: '))
        img = postprocess_black(original_img, boxes, masks, idx)
        fileName = args.image[:-4] + '_black.jpg'
        cv.imwrite(fileName, img.astype(np.uint8))

# Extract the bounding box and mask for each detected object
def postprocess_segment(image, boxes, masks):
    # Output size of masks is NxCxHxW where
    # N - number of detected boxes
    # C - number of classes (excluding background)
    # HxW - segmentation shape
    numClasses = masks.shape[1]
    numDetections = boxes.shape[2]

    imageH = image.shape[0]
    imageW = image.shape[1]

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

            # Extract the mask for the object
            classMask = mask[classId]

            # Draw bounding box, colorize and show the mask on the image
            drawBox(image, i, left, top, right, bottom, classMask)

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
def drawBox(frame, idx, left, top, right, bottom, classMask):
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
    cv.drawContours(diff[top:bottom + 1, left:right + 1], contours, -1, color=(0, 0, 0), thickness=cv.FILLED)
    image = cv.subtract(image, diff)

    return image

if __name__ == '__main__':
    main()
