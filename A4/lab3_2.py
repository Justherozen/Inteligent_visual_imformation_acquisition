import cv2
import numpy as np

left_camera_matrix = np.array([[749.43878642,   0.,         251.26052903],
 [  0.,         747.15393015, 191.76971714],
 [  0.,           0.,           1.        ]]
)
left_distortion = np.array([-0.54115607, -0.14901452,  0.00562021,  0.00230344,  1.40562318])

right_camera_matrix = np.array([[737.72932697,   0.,         249.49842471],
 [  0.,         736.60284178, 198.14313865],
 [  0.,           0.,           1.        ]])
right_distortion = np.array([-4.92326558e-01, -3.89038900e-01, -7.62771755e-05,  3.92085144e-03, 2.45869839e+00])
scale_size = 5
cb_width = 7
cb_height = 6
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((cb_width*cb_height, 3), np.float32)
objp[:, :2] = np.mgrid[:cb_width, :cb_height].T.reshape(-1, 2)
objpoint = []
objpoint.append(objp)
imgpointsR = []
imgpointsL = []
img1 = cv2.imread("./cl3.jpg")
img2 = cv2.imread("./cr3.jpg")
img3 = cv2.imread("./cl3.jpg")
img4 = cv2.imread("./cr3.jpg")

#img1 = cv2.resize(img1, (img1.shape[1]//scale_size, img1.shape[0]//scale_size))
#img2 = cv2.resize(img2, (img2.shape[1]//scale_size, img2.shape[0]//scale_size))
#img3 = cv2.resize(img3, (img3.shape[1]//scale_size, img3.shape[0]//scale_size))
#img4 = cv2.resize(img4, (img4.shape[1]//scale_size, img4.shape[0]//scale_size))
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
ret, corners1 = cv2.findChessboardCorners(gray1, (cb_width, cb_height), None)
if ret == True:
    print("1")
    rt = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
    imgpointsL.append(corners1)

gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret, corners2 = cv2.findChessboardCorners(gray2, (cb_width, cb_height), None)
if ret == True:
    print("2")
    rt = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
    imgpointsR.append(corners2)

flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
flags |= cv2.CALIB_SAME_FOCAL_LENGTH
flags |= cv2.CALIB_FIX_FOCAL_LENGTH
flags |= cv2.CALIB_FIX_ASPECT_RATIO
flags |= cv2.CALIB_FIX_K1
flags |= cv2.CALIB_FIX_K2
flags |= cv2.CALIB_FIX_K3
flags |= cv2.CALIB_FIX_K4
flags |= cv2.CALIB_FIX_K5

stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
ret, M1, d1, M2, d2, R, T, E, F = \
    cv2.stereoCalibrate(objpoint, imgpointsL, imgpointsR, left_camera_matrix,
                        left_distortion, right_camera_matrix, right_distortion,
                        criteria=stereocalib_criteria, flags=flags,
                        imageSize=(img1.shape[1], img1.shape[0]))

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = \
    cv2.stereoRectify(left_camera_matrix, left_distortion, right_camera_matrix,
                      right_distortion, (img1.shape[1], img1.shape[0]), R, T, flags=1)
left_map1, left_map2 = cv2.initUndistortRectifyMap(\
    left_camera_matrix, left_distortion, R1, P1,
    (img3.shape[1], img3.shape[0]), cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(\
    right_camera_matrix, right_distortion, R2, P2,
    (img4.shape[1], img4.shape[0]), cv2.CV_16SC2)
re_img1 = cv2.remap(img3, left_map1, left_map2, cv2.INTER_LINEAR)
re_img2 = cv2.remap(img4, right_map1, right_map2, cv2.INTER_LINEAR)
#cv2.imshow("_left", img1)
#cv2.imshow("_right", img2)
cv2.imwrite("./cal_left.jpg", re_img1)
cv2.imwrite("./cal_right.jpg", re_img2)
k = cv2.waitKey(0) & 0xff
if k == 27:
    cv2.destroyAllWindows()
