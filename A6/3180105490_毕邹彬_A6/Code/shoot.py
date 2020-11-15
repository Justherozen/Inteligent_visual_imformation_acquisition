import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from pypylon import pylon

camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

scale_size = 5

i = 1
flag = False
while camera.IsGrabbing():

    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        image = converter.Convert(grabResult)
        img = image.GetArray()
        cv2.namedWindow('title', cv2.WINDOW_NORMAL)
        cv2.imshow('title', img)
        k = cv2.waitKey(300)
        if k == 27:
            break
        if k == 13:
            flag = ~flag
        if flag:
            print("capturing...")
            a = img.shape[1] - img.shape[0]
            a = int(a / 2)
            a = a - 100
            cv2.imwrite("./pro_data/" + str(i) + ".jpg", img[0:img.shape[0], a:a + img.shape[0]])
            i = i + 1
        else:
            print("waiting...")
    grabResult.Release()
camera.StopGrabbing()
cv2.destroyAllWindows()
