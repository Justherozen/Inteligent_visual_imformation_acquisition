import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob

def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    # draw pillars in blue color
    for i,j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    return img

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)
axis = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0],
                   [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])
# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob('./chessboard/*.jpg')

scale_size = 6
rotMat = np.zeros((3, 3))
camera_center = np.zeros((3, 1))
i = 0
f = open("date1.ply", 'w')

for fname in images:
    img = cv2.imread(fname)
    img = cv2.resize(img, (img.shape[1]//scale_size, img.shape[0]//scale_size))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    what = ret, corners = cv2.findChessboardCorners(gray, (7, 7), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        # cv2.imshow(fname+"Orig", cv2.resize(img, (img.shape[1]//4, img.shape[0]//4)))
        plt.subplot(1, 2, 1), plt.imshow(img), plt.title(fname+" Orig")

        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)

        imgp = imgpoints[i][:, 0]
        objp = np.array(objpoints)[i]
        cv2.Rodrigues(np.array(rvecs[i]), rotMat)
        camera_postion =np.dot(rotMat.T, - tvecs[i])

        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs[i], tvecs[i], mtx, dist)
        img = draw(img, corners2, imgpts)
        plt.subplot(1, 2, 2), plt.imshow(img), plt.title(fname+" AR")
        plt.show()
        cv2.waitKey(500)
        i = i + 1

f.close()
k = cv2.waitKey(0) & 0xff
if k == 27:
    cv2.destroyAllWindows()
