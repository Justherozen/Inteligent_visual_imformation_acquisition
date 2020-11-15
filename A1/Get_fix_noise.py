# 导入cv模块
import cv2 as cv
import numpy as np
img = cv.imread("./9000.png")
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = cv.bilateralFilter(img, 3, 2, 0)
#img_avg = cv.imread("./avgpic/3000.png", cv.IMREAD_GRAYSCALE)
#img = cv.subtract(img, img_avg)
th, img = cv.threshold(img, 4, 255, cv.THRESH_BINARY)
cv.imwrite("./20.png", img)
#cv.namedWindow("image")
#cv.imshow("image", img)
cv.waitKey(0)
cv.destroyAllWindows()
