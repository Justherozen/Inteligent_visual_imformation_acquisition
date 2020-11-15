# 导入cv模块
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
exposure=[2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000,70000,120000,170000,220000]
avgnoise_exp=[]
for j in range(0,len(exposure)):
    print("./fixgain/"+str(exposure[j])+"(0).jpg")
    raw_img = cv.imread("./fixgain/2000(0).jpg")
    size = raw_img.shape
    print("!")
    imgarray = np.zeros([10, size[0]*size[1]], dtype=np.float)
    for i in range(0, 10):
        img = cv.imread("./fixgain/"+str(exposure[j])+"("+ str(i) +").jpg")
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = img.astype(np.float32)
        imgarray[i, :] = img.flatten()
    avgarray = np.mean(imgarray, 0)
    std_noise=np.std(avgarray)
    avgnoise_exp.append(std_noise)
plt.plot(exposure,avgnoise_exp,linewidth=2)
plt.title("fix pattern noise-exp",fontsize=18)
plt.xlabel("exposure",fontsize=14)
plt.ylabel("avg_noise",fontsize=14)
plt.savefig("./fpnoise_exp.png")