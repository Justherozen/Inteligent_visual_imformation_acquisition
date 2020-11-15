import numpy as np
import matplotlib.pyplot as plt
import cv2

scale_size = 2
cb_width = 7
cb_height = 7

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((cb_width*cb_height, 3), np.float32)
objp[:, :2] = np.mgrid[:cb_width, :cb_height].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
i = 0
rotMat = np.zeros((3, 3))

images = ['center1', 'center2', 'center3', 'center4', 'c_left', 'c_right', 'extra',
          'extra2', 'extra3', 'extra4', 'left1', 'right', 'right_up', 'right_up2', 'up']


print("| Photo | Total Error|\n| --- | --- |")

for fname in images:
    img = cv2.imread('./chessboard/'+fname+'.jpg')
    img = cv2.resize(img, (img.shape[1]//scale_size, img.shape[0]//scale_size))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (cb_width, cb_height), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1), plt.imshow(img), plt.title(fname+" Original")
        orig_h, orig_w = img.shape[:2]

        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)

        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            mtx, dist, (orig_w, orig_h), 1, (orig_w, orig_h))

        imgp = imgpoints[i][:, 0]
        objp = np.array(objpoints)[i]
        cv2.Rodrigues(np.array(rvecs[i]), rotMat)
        camera_postion = np.dot(rotMat.T, - tvecs[i])
        total_error = 0
        for j in range(len(objp)):
            imgpoints2, _ = cv2.projectPoints(objp[j], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgp[j], imgpoints2[0][0], cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        print("|", fname, "|", total_error / len(objpoints), "|")
        plt.savefig('./calibrated/'+fname+'.jpg', format='jpg')
        i += 1
        cv2.waitKey(500)

k = cv2.waitKey(0) & 0xff
if k == 27:
    cv2.destroyAllWindows()
