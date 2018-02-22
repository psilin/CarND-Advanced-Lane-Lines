
import cv2

import helpers
import test_utils

def run():
    print('hello world!')

    path = './../camera_cal/'

    ret, mtx, dist = helpers.calibrateCamera(path)
    test_utils.testCameraCalibration(path, ret, mtx, dist)



if __name__ == '__main__':
    run()

