
import cv2

import helpers
import test_utils

def run():
    ret, mtx, dist = helpers.calibrateCamera('./../camera_cal/')
    #test_utils.testCameraCalibration(path, mtx, dist)
    M, Minv = helpers.warp_matrix()
    #test_utils.testWarping('./../test_images', M, mtx, dist)
    test_utils.testSobelHLS('./../test_images', M, mtx, dist)


if __name__ == '__main__':
    run()

