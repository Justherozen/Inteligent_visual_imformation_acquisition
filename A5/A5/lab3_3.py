from __future__ import print_function
import sys
import numpy as np
import cv2
import cv2 as cv
import matplotlib.pyplot as plt
ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    invaild = 0
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

scale_size = 5

def main():
    print('loading images...')
    imgL = cv.pyrDown(cv.imread(cv.samples.findFile("./cl3.jpg")))  # downscale images for faster processing
    imgR = cv.pyrDown(cv.imread(cv.samples.findFile("./cr3.jpg")))
    #imgR = cv2.resize(imgL, (imgR.shape[1]//scale_size, imgR.shape[0]//scale_size))
    #imgL = cv2.resize(imgL, (imgL.shape[1]//scale_size, imgL.shape[0]//scale_size))
    height, width = imgR.shape[:2]
    # imgL = cv.pyrDown(cv.imread(cv.samples.findFile("./lab3_data/cl5.bmp")))  # downscale images for faster processing
    # imgR = cv.pyrDown(cv.imread(cv.samples.findFile("./lab3_data/cr5.bmp")))
    window_size = 3  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=-1,
        numDisparities=5*16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=window_size,
        P1=8 * 3 * window_size,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size,
        disp12MaxDiff=12,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 80000
    sigma = 1.3
    visual_multiplier = 6

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)

    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)
    cv.imshow('left', filteredImg)
    cv.imwrite("left_depth6.jpg", filteredImg)
    h, w = imgL.shape[:2]
    f = 0.8 * w
    Q=np.float32([[1, 0, 0, -0.5*w],
				[0, -1, 0, 0.5*h],
				[0, 0, 0, -f], #Focal length multiplication obtained experimentally.
				[0, 0, 1, 0]])
    points = cv.reprojectImageTo3D(displ.astype(np.float32), Q)
    np.set_printoptions(threshold=sys.maxsize)
    haha=imgL
    colors = cv.cvtColor(imgL, cv.COLOR_BGR2RGB,dst=haha)

    points=points.reshape(-1,3)
    colors=colors.reshape(-1,3)

    fpoints=None
    fcolors=None
    '''
    #filter
    for i in range(0,colors.shape[0]):
        if (colors[i, 0]<150 and colors[i, 1]<150 and colors[i, 1]<150):
            if fpoints is None:
                fpoints=points[i]
                fcolors=colors[i]
            else:
                fpoints=np.vstack((fpoints,points[i]))
                fcolors=np.vstack((fcolors,colors[i]))
    '''
    fpoints = points
    fcolors = colors

    out_fn = 'out.ply'
    write_ply(out_fn, fpoints, fcolors)
    print('Done')
    cv.waitKey()


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()