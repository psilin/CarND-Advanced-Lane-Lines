
import cv2
import numpy as np
import os


def calibrateCamera(path):
    """
    @brief all camera calibration related stuff
    """
    nx = 9
    ny = 6
    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    objpoints = []
    imgpoints = []

    shape = None
    if (os.path.isdir(path)):
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if not os.path.isfile(file_path):
                continue

            img = cv2.imread(file_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            shape = gray.shape[::-1]
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)

    return ret, mtx, dist


def warp_matrix():
    """
    @brief get warp matrix
    """
    p1 = (315, 650)
    p2 = (1005, 650)
    p3 = (525, 500)
    p4 = (765, 500)

    src = np.float32([p1, p2, p3, p4])
    dst = np.float32([p1, p2, (p1[0], 360), (p2[0], 360)])

    img_size = (1280, 720)
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv
