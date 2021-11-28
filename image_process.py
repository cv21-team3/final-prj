# bitwise_and 연산으로 마스킹하기 (bitwise_masking.py)

import numpy as np
import cv2
import matplotlib.pylab as plt

#--① 이미지 읽기
img = cv2.imread('case8_raw.png')

#--② 마스크 만들기
mask = cv2.imread('case8_mask_before.png', flags=-1)
print(mask.shape)
#print(img[0,0,:])
#mask = cv2.resize(mask, (0,0), fx=0.55, fy=0.55)

for y in range(mask.shape[1]):
    for x in range(mask.shape[0]):
        if mask[x,y,0] == 255 and mask[x,y,1] == 255 and mask[x,y,2] == 255:
            mask[x,y,0] = 255
            mask[x,y,1] = 255
            mask[x,y,2] = 255
            img[x,y,0] = 255
            img[x,y,1] = 255
            img[x,y,2] = 255
        else:
            mask[x,y,0] = 0
            mask[x,y,1] = 0
            mask[x,y,2] = 0

cv2.imwrite('case8_mask.png', mask)
cv2.imwrite('case8_input.png', img)

#cv2.circle(mask, (260,210), 100, (255,255,255), -1)
#cv2.circle(대상이미지, (원점x, 원점y), 반지름, (색상), 채우기)

# #--③ 마스킹
# masked = cv2.bitwise_and(img, mask)

# #--④ 결과 출력
# cv2.imshow('original', img)
# cv2.imshow('mask', mask)
# cv2.imshow('masked', masked)
# cv2.waitKey()
# cv2.destroyAllWindows()