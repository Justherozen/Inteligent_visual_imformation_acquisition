from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import cv2

img_path = "./colorD/"
img_L = ['l1', 'l2', 'l3', 'l4']
img_R = ['r1', 'r2', 'r3', 'r4']
img_format = ".bmp"

img_redl = cv2.imread('./colorD/redl.bmp')
img_redr = cv2.imread('./colorD/redr.bmp')
img_greenl = cv2.imread('./colorD/greenl.bmp')
img_greenr = cv2.imread('./colorD/greenr.bmp')
img_bluel = cv2.imread('./colorD/bluel.bmp')
img_bluer = cv2.imread('./colorD/bluer.bmp')

max_redl = cv2.mean(img_redl)[2]
max_redr = cv2.mean(img_redr)[2]
max_greenl = cv2.mean(img_greenl)[1]
max_greenr = cv2.mean(img_greenr)[1]
max_bluel = cv2.mean(img_bluel)[0]
max_bluer = cv2.mean(img_bluer)[0]

for img_name in img_L:
    img = cv2.imread(img_path+img_name+img_format).astype('float64')
    img[:, :, 0] *= 255 / max_bluel
    img[:, :, 1] *= 255 / max_greenl
    img[:, :, 2] *= 255 / max_redl
    cv2.imwrite("./colorDed/"+img_name+img_format, img)

for img_name in img_R:
    img = cv2.imread(img_path+img_name+img_format).astype('float64')
    img[:, :, 0] *= 255 / max_bluer
    img[:, :, 1] *= 255 / max_greenr
    img[:, :, 2] *= 255 / max_redr
    cv2.imwrite("./colorDed/"+img_name+img_format, img)

exit(0)
