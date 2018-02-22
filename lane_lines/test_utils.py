
import cv2
import helpers
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

def testCameraCalibration(path, mtx, dist):
    """
    @brief test camera calibration
    """
    if (os.path.isdir(path)):
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if not os.path.isfile(file_path):
                continue

            img = cv2.imread(file_path)
            undistorted = cv2.undistort(img, mtx, dist, None, mtx)
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
            f.tight_layout()
            ax1.imshow(img)
            ax1.set_title('Original Image', fontsize=50)
            ax2.imshow(undistorted)
            ax2.set_title('Undistorted Image', fontsize=50)
            plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
            plt.show()


def testWarping(path, M, mtx, dist):
    """
    @brief test images warping given test images dir
    """
    if (os.path.isdir(path)):
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if not os.path.isfile(file_path):
                continue

            img = cv2.imread(file_path)
            undistorted = cv2.undistort(img, mtx, dist, None, mtx)
            undistorted = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
            p1 = (315, 650)
            p2 = (1005, 650)
            p3 = (525, 500)
            p4 = (765, 500)
            cv2.line(undistorted, p1, p2, (255, 0, 0), 5)
            cv2.line(undistorted, p3, p4, (255, 0, 0), 5)
            cv2.line(undistorted, p1, p3, (255, 0, 0), 5)
            cv2.line(undistorted, p2, p4, (255, 0, 0), 5)

            gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
            img_size = (gray.shape[1], gray.shape[0])
            warped = cv2.warpPerspective(gray, M, img_size, flags=cv2.INTER_LINEAR)

            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
            f.tight_layout()
            ax1.imshow(undistorted, cmap='gray')
            ax1.set_title('Original Undistorted Image', fontsize=50)
            ax2.imshow(warped, cmap='gray')
            ax2.set_title('Warped Image', fontsize=50)
            plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
            plt.show()


def testSobelHLS(path, M, mtx, dist):
    """
    @brief test sobel and HLS application
    """
    if (os.path.isdir(path)):
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if not os.path.isfile(file_path):
                continue

            img = cv2.imread(file_path)
            undistorted = cv2.undistort(img, mtx, dist, None, mtx)
            rgb = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
            img_size = (rgb.shape[1], rgb.shape[0])
            warped = cv2.warpPerspective(rgb, M, img_size, flags=cv2.INTER_LINEAR)
            combined_binary, color_binary = helpers.apply_sobel_and_hls(warped)

            f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 9))
            f.tight_layout()
            ax1.imshow(rgb, cmap='gray')
            ax1.set_title('Original undistorted image', fontsize=10)
            ax2.imshow(warped, cmap='gray')
            ax2.set_title('Warped image', fontsize=10)
            ax3.imshow(color_binary)
            ax3.set_title('Stacked thresholds', fontsize=10)
            ax4.imshow(combined_binary, cmap='gray')
            ax4.set_title('S channel and gradient threshold combined', fontsize=10)
            plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0.05)
            plt.show()

