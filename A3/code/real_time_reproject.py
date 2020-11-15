import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from pypylon import pylon
from objloader_simple import *

obj = OBJ(('./models/fox.obj'), swapyz=True)
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:6, 0:7].T.reshape(-1, 2)
axis = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0],
                   [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])
mtx = np.array([[626.23949494, 0, 211.21608234],
                [0, 626.22136938, 162.64737962],
                [0., 0., 1.]])
dist = np.array([[-5.82531846e-01, 2.49567823e-01, 3.59964573e-03, -1.10895344e-04
                     , -3.18090306e-01]])
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
rvecs = []
tvecs = []
images = glob.glob('.essboard/*.jpg')

scale_size = 5
rotMat = np.zeros((3, 3))
camera_center = np.zeros((3, 1))
i = 0
f = open("date1.ply", 'w')
while camera.IsGrabbing():
    i = i + 1
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        image = converter.Convert(grabResult)
        img = image.GetArray()
        cv2.namedWindow('title', cv2.WINDOW_NORMAL)
        cv2.imshow('title', img)
        k = cv2.waitKey(1)
        if k == 27:
            break

        img = cv2.resize(img, (img.shape[1] // scale_size, img.shape[0] // scale_size))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(str(i) + ".jpg", img)
        what = ret, corners = cv2.findChessboardCorners(gray, (6, 7), None)
        if ret == True:
            print("found chessboard")
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None)

            h, w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                mtx, dist, (w, h), 1, (w, h))
            vertices = obj.vertices
            scale_matrix = np.eye(3) * 0.02
            h = img.shape[0]
            w = img.shape[1]
            color = False
            for face in obj.faces:
                face_vertices = face[0]
                points = np.array([vertices[vertex - 1] for vertex in face_vertices])
                points = np.dot(points, scale_matrix)
                points[:, 2] = points[:, 2] * (-1)
                imgpts, jac = cv2.projectPoints(points, rvecs, tvecs, mtx, dist)
                imgpts = np.int32(imgpts).reshape(-1, 2)
                if color is False:
                    cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
                else:
                    color = hex_to_rgb(face[-1])
                    color = color[::-1]
                    cv2.fillConvexPoly(img, imgpts, color)
            cv2.imshow("demo", img)
            cv2.waitKey(10)
            i = i + 1
    grabResult.Release()
camera.StopGrabbing()
f.close()
k = cv2.waitKey(0) & 0xff
if k == 27:
    cv2.destroyAllWindows()
