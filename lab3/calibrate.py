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

images = ['center1', 'center2', 'center3', 'center4', 'c_left', 'c_right', 'extra',
          'extra2', 'extra3', 'extra4', 'left1', 'right', 'right_up', 'right_up2', 'up']


for fname in images:
    img = cv2.imread('./raw/'+fname+'.jpg')
    img = cv2.resize(img, (img.shape[1]//scale_size, img.shape[0]//scale_size))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (cb_width, cb_height), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1), plt.imshow(img), plt.title(fname+" Original")

        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)

        h,  w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            mtx, dist, (w, h), 1, (w, h))

        # undistort
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        plt.subplot(1, 2, 2), plt.imshow(dst), plt.title(fname+" Calibrated")
        plt.savefig('./calibrated/'+fname+'.jpg', format='jpg')
        # plt.show()
        cv2.waitKey(500)


k = cv2.waitKey(0) & 0xff
if k == 27:
    cv2.destroyAllWindows()
