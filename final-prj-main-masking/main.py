import os
import math
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from facial_landmarks import *
from segmentor import *
import cv2 as cv
import argparse
import os.path
import sys
import random
from imutils import face_utils
import imutils
import dlib

def select_target(args):
    global classes, colors, boxes, masks, original_img

    ############################
    # step 1: find removal target
    #  ㄴ1-1: segmentation
    ###########################

    # Initialize the parameters
    confThreshold = 0.5  # Confidence threshold
    maskThreshold = 0.1  # Mask threshold
    classToDetect = ['person'] # Classes to detect

    classes = None
    colors = []

    # ==================== Load names of classes  ====================
    classesFile = "mscoco_labels.names"
    with open(classesFile, 'rt') as f:
      classes = f.read().rstrip('\n').split('\n')

    # ==================== Load the segmentation network ====================
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

    #winName = 'Object detection'
    #cv.namedWindow(winName, cv.WINDOW_NORMAL)

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
    box_size, bbox_coord=postprocess_segment(classes,colors,img, boxes, masks)  # Extract the bounding box and mask for each of the detected objects
    #uncomment these two lines if you want to save segmentation result
    #fileName = args.image[:-4] + '_segmented.jpg'
    #cv.imwrite(fileName, img.astype(np.uint8))
    ######################
    # step 1: find removal target
    #  ㄴ1-2: facial detection
    ######################
  
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
    image = original_img.copy()

    complete_landmarks_detected_img, facial_coord= show_raw_detection(image, detector, predictor)
    #uncomment these two lines if you want to save face detection result
    #fileName = args.image[:-4] + '_face_detected.jpg'
    #cv.imwrite(fileName, complete_landmarks_detected_img.astype(np.uint8))

    ######################
    # step 1: find removal target
    #  ㄴ1-3: choose removal target
    ######################

    fdetected_i=[]
    fdetected=[]
    for i in range(len(facial_coord)):
      for j in range(len(bbox_coord)):
        if (bbox_coord[j][0]<=facial_coord[i][0]) and (facial_coord[i][0]+facial_coord[i][2]<=bbox_coord[j][2]) and (bbox_coord[j][1]<=facial_coord[i][1]) and (facial_coord[i][1]+facial_coord[i][3]<=bbox_coord[j][3]):
          fdetected_i.append(j)
          fdetected.append(box_size[j])
    target=np.argmax(box_size)
    if len(facial_coord)>1:
      if len(fdetected)!=0:  
        target=fdetected_i[np.argmax(fdetected)]
      else:
        target=np.argmax(box_size)
    elif len(facial_coord)==1:
      if len(fdetected)!=0:
        target=fdetected_i[0]
    return target,bbox_coord


def mask(args,target,bbox_coord):
    global classes,colors,masks,boxes,original_img

    image=original_img.copy()
    masked_image=image.copy()
    temp=image.copy()
    masked_traces=np.empty_like(image)
    for i in range(len(bbox_coord)):
      if i!=target:
        temp=postprocess_black(masked_image,boxes,masks,i)
        masked_image=cv.subtract(masked_image,temp)
    masked_traces=cv.subtract(image,masked_image)
    fileName = args.image[:-4] + 'masked.jpg'
    cv.imwrite(fileName, masked_image.astype(np.uint8))
    fileName = args.image[:-4] + 'masked_traces.jpg'
    cv.imwrite(fileName, masked_traces.astype(np.uint8))
    return masked_image,masked_traces
# masked_image : masking 된 image
# masked_traces : masking한 조각들이 모여있는 image

def repaint(igs_in):

    return igs_in

def main():
    # read img

    global classes, colors,masks,original_img
    parser = argparse.ArgumentParser(description='image preprocess')
    parser.add_argument('--image', help='Path to image file')
    args = parser.parse_args()

    ##############
    # step 1: selecting removal target region
    ##############
    target,bbox_coord = select_target(args)

    ##############
    # step 2: masking removal target region
    ##############
    masked_img, masked_traces = mask(args,target,bbox_coord)

    ##############
    # step 3: repainting masked region
    ##############
    #repainted = repaint(masked)
    #Image.fromarray(repainted.astype(np.uint8)).save('data/result/(예시)1-2.png')

if __name__ == '__main__':
    main()
