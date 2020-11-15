# 导入cv模块
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
exposure=[2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000,70000,120000,170000,220000]
avgnoise_exp=[]
for j in range(0,len(exposure)):
    raw_img = cv.imread("./fixgain/"+str(exposure[j])+"(0).jpg")
    size = raw_img.shape
    #img=cv.resize(img, shrink,interpolation=cv.INTER_CUBIC)
    #img=cv.resize(raw_img, (0, 0), fx=0.25, fy=0.25, interpolation=cv.INTER_NEAREST)    
    #cv.namedWindow("image")
    
    imgarray = np.zeros([10, size[0]*size[1]], dtype=np.float)
    for i in range(0, 10):
        #shrink = (int(width*0.3), int(height*0.3))  
    #img=cv.resize(img, shrink,interpolation=cv.INTER_CUBIC)
        #img=cv.resize(raw_img, (0, 0), fx=0.25, fy=0.25, interpolation=cv.INTER_NEAREST)  
        img = cv.imread("./fixgain/"+str(exposure[j])+"("+ str(i) +").jpg")
        #img = cv.resize(img, (size[0], size[1]))å
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = img.astype(np.float32)
        imgarray[i, :] = img.flatten()
    std_noise = np.std(imgarray, 0)
    #cv.imshow("image", std_noise.reshape(size[1], size[0]).astype(np.uint8)*10)
    avgarray = np.mean(imgarray, 0)
    imgavgarray = avgarray.reshape(size[0], size[1])
    #median_noise = np.median(std_noise, 0)
    median_noise = np.mean(std_noise)
    avgnoise_exp.append(median_noise)
    plt.scatter(avgarray, std_noise,s=0.1, alpha=0.6) 
    plt.title("noise-avgsignal,"+"exposure t="+str(exposure[j]),fontsize=18)
    plt.xlabel("avg_signal",fontsize=14)
    plt.ylabel("noise",fontsize=14) 
    plt.savefig("./signalnoiseratio/"+str(exposure[j])+".png")
    plt.close()
    #plt.show()
    cv.imwrite("./avgpic/"+str(exposure[j])+".png", imgavgarray.astype(np.uint8))
    #cv.imshow("image", imgavgarray.astype(np.uint8))
    cv.waitKey(0)
    cv.destroyAllWindows()
plt.plot(exposure,avgnoise_exp,linewidth=2)
plt.title("noise-exposuret",fontsize=18)
plt.xlabel("exposure time",fontsize=14)
plt.ylabel("avg_noise",fontsize=14)
plt.savefig("./noise_exposure.png")