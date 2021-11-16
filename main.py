import os
import math
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def select_removal_target(igs_in):
    for x in range(igs_in.shape[0]):
        if x not in range(140, 322):
            igs_in[x, :, :] = 0
        else:
            for y in range(igs_in.shape[1]):
                if y not in range(210, 270):
                    igs_in[x, y, :] = 0

    if igs_in is not None:
        plt.figure()
        plt.imshow(igs_in.astype(np.uint8))
        plt.axis()
        plt.show()

    return igs_in

def mask(igs_in):
    for x in range(180, 278):
        for y in range(350, 400):
            igs_in[x, y, :] = 0

    if igs_in is not None:
        plt.figure()
        plt.imshow(igs_in.astype(np.uint8))
        plt.axis()
        plt.show()

    return igs_in

def repaint(igs_in):

    return igs_in

def main():
    # read img
    img_in = Image.open('data/raw/(예시)1-2.png').convert('RGB')
    igs_in = np.array(img_in)

    print("img_in.shape: ", igs_in.shape, "\n")

    ##############
    # step 1: selecting removal target region
    ##############
    target = select_removal_target(igs_in.copy())
    Image.fromarray(target.astype(np.uint8)).save('data/target/(예시)1-2.png')

    ##############
    # step 2: masking removal target region
    ##############
    masked = mask(igs_in.copy())
    Image.fromarray(masked.astype(np.uint8)).save('data/masked/(예시)1-2.png')

    ##############
    # step 3: repainting masked region
    ##############
    repainted = repaint(masked)
    Image.fromarray(repainted.astype(np.uint8)).save('data/result/(예시)1-2.png')

if __name__ == '__main__':
    main()
