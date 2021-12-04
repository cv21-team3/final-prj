import os
import numpy as np
import cv2
from scipy.interpolate import RectBivariateSpline
from skimage.filters import apply_hysteresis_threshold


def lucas_kanade_affine(img1, img2, p, Gx, Gy):
    ### START CODE HERE ###
    # [Caution] From now on, you can only use numpy and
    # RectBivariateSpline. Never use OpenCV.
    img1 = img1.astype(float)
    img2 = img2.astype(float)

    h = img1.shape[0]
    w = img1.shape[1]
    M = np.array(  # Affine transform matrix
        [
            [1 + p[0, 0], p[1, 0], p[4, 0]],
            [p[2, 0], 1 + p[3, 0], p[5, 0]]
        ]
    )

    H = np.zeros((6, 6), dtype=float)  # Hessian
    err = np.zeros((6, 1), dtype=float)  # Matrix to multiply to Hessian
    gradient = np.array([[0, 0]], dtype=float)
    J = np.array([[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]], dtype=float)
    diff = np.array([[0]], dtype=float)
    for y in range(h):
        for x in range(w):
            # Warp the coordinates
            warped_coord = M @ np.array([[x], [y], [1]])
            warped_x = int(warped_coord[0, 0])
            warped_y = int(warped_coord[1, 0])

            if 0 <= warped_y < img2.shape[0] and 0 <= warped_x < img2.shape[1]:
                gradient[0, 0] = Gx[warped_y, warped_x]
                gradient[0, 1] = Gy[warped_y, warped_x]
                J[0, 0] = x
                J[0, 2] = y
                J[1, 1] = x
                J[1, 3] = y
                mul = gradient @ J  # 1 x 6

                H += mul.transpose() @ mul  # Compute Hessian
                diff[0, 0] = img1[y, x] - img2[warped_y, warped_x]
                err += mul.transpose() @ diff  # Compute the error matrix

    H_inv = np.linalg.inv(H)
    dp = H_inv @ err

    ### END CODE HERE ###
    return dp


def subtract_dominant_motion(img1, img2):
    Gx = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=5)  # do not modify this
    Gy = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=5)  # do not modify this

    ### START CODE HERE ###
    # [Caution] From now on, you can only use numpy and
    # RectBivariateSpline. Never use OpenCV.

    th_hi = 0.2 * 256  # you can modify this
    th_lo = 0.15 * 256  # you can modify this

    p = np.array([[0, 0, 0, 0, 0, 0]], dtype=float).transpose()
    for i in range(12):
        dp = lucas_kanade_affine(img1, img2, p, Gx, Gy)
        p += dp
        if np.linalg.norm(dp) < 0.01:
            break

    M = np.array(  # Affine transform matrix
        [
            [1 + p[0, 0], p[1, 0], p[4, 0]],
            [p[2, 0], 1 + p[3, 0], p[5, 0]],
            [0, 0, 1]
        ]
    )

    warped_img1 = warp(M, img1)
    moving_image = img2 - warped_img1
    ### END CODE HERE ###

    hyst = apply_hysteresis_threshold(moving_image, th_lo, th_hi)
    return hyst

def get_affine(img1, img2):
    Gx = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=5)  # do not modify this
    Gy = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=5)  # do not modify this

    p = np.array([[0, 0, 0, 0, 0, 0]], dtype=float).transpose()
    for i in range(10):
        dp = lucas_kanade_affine(img1, img2, p, Gx, Gy)
        p += dp
        if np.linalg.norm(dp) < 0.01:
            break

    M = np.array(  # Affine transform matrix
        [
            [1 + p[0, 0], p[1, 0], p[4, 0]],
            [p[2, 0], 1 + p[3, 0], p[5, 0]],
            [0, 0, 1]
        ]
    )

    return M

def warp(M, img):
    h = img.shape[0]
    w = img.shape[1]
    warped = np.zeros(img.shape)
    M_inv = np.linalg.inv(M)
    for y in range(warped.shape[0]):
        for x in range(warped.shape[1]):
            coord = M_inv @ np.array([[x], [y], [1]])
            warped_x = int(coord[0, 0])
            warped_y = int(coord[1, 0])
            if 0 <= warped_y < h and 0 <= warped_x < w:
                warped[y, x] = img[warped_y, warped_x]
    return warped


if __name__ == "__main__":
    data_dir = 'data'
    video_path = 'motion.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 150 / 20, (636, 318))
    tmp_path = os.path.join(data_dir, "organized-{}.jpg".format(0))
    T = cv2.cvtColor(cv2.imread(tmp_path), cv2.COLOR_BGR2GRAY)
    for i in range(0, 50):
        img_path = os.path.join(data_dir, "organized-{}.jpg".format(i))
        I = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
        clone = I.copy()
        moving_img = subtract_dominant_motion(T, I)
        clone = cv2.cvtColor(clone, cv2.COLOR_GRAY2BGR)
        clone[moving_img, 2] = 522
        out.write(clone)
        T = I
    out.release()
