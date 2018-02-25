
import cv2
import numpy as np
import os

class LowPassFilter:
    """
    @brief low pass filter for polynomes
    """
    def __init__(self):
        self.init = False
        self.alpha = 1./8
        self.left = None
        self.right = None


    def run(self, left_fit, right_fit):
        if self.init is False:
            self.left = left_fit
            self.right = right_fit
            self.init = True
        else:
            self.left = self.left * (1. - self.alpha) + left_fit * self.alpha
            self.right = self.right * (1. - self.alpha) + right_fit * self.alpha

        return self.left, self.right


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


def apply_mask(img):
    """
    @brief select region where lane line is
    """
    ysize = img.shape[0]
    xsize = img.shape[1]
    region_select = np.copy(img)

    x1 = (100, 590)
    y1 = (720, 450)
    x2 = (750, 1200)
    y2 = (450, 720)

    #ban outside of trapeze
    fit_left = np.polyfit(x1, y1, 1)
    fit_right = np.polyfit(x2, y2, 1)

    x3 = (300, 650)
    y3 = (720, 450)
    x4 = (1000, 650)
    y4 = (720, 450)

    #ban inside of triangle
    fit_inner_left = np.polyfit(x3, y3, 1)
    fit_inner_right = np.polyfit(x4, y4, 1)

    # Find the region outside the lines
    XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
    outter_region_thresholds = (YY < 450) | (YY < (XX*fit_left[0] + fit_left[1])) | (YY < (XX*fit_right[0] + fit_right[1])) | ((YY > (XX*fit_inner_left[0] + fit_inner_left[1])) & (YY > (XX*fit_inner_right[0] + fit_inner_right[1])))
    region_select[outter_region_thresholds] = 0#[0, 0, 0]
    return region_select


def apply_sobel_and_hls(img):
    """
    @brief apply SobelX operator and HLS to image
    """

    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]

    # Sobel x
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Threshold color channel
    s_thresh_min = 170
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary, color_binary


def get_curvature(binary_warped, left_fit, right_fit):
    """
    @brief compute curvature radius of left and right lale lines
    """
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    y_eval = np.max(ploty)

    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    return left_curverad, right_curverad


def window_search(binary_warped, low_pass):
    """
    @brief apply window search to find lane on binary warped image
    """
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    #filter polys
    if (low_pass is not None):
        left_fit, right_fit = low_pass.run(left_fit, right_fit)

    #compute curvature of lane lines in meters
    left_curve_rad, right_curve_rad = get_curvature(binary_warped, left_fit, right_fit)

    #compute car center offset from middle of the lane
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    y = out_img.shape[0]
    leftx = int(round(left_fit[0]*y**2 + left_fit[1]*y + left_fit[2]))
    rightx = int(round(right_fit[0]*y**2 + right_fit[1]*y + right_fit[2]))
    center_offset = (out_img.shape[1] - leftx - rightx)//2 * xm_per_pix

    lane_img = np.zeros_like(out_img)
    for y in range(0, lane_img.shape[0]):
        leftx = int(round(left_fit[0]*y**2 + left_fit[1]*y + left_fit[2]))
        rightx = int(round(right_fit[0]*y**2 + right_fit[1]*y + right_fit[2]))
        lane_img[y,leftx:rightx + 1] = [0, 255, 0]

    return lane_img, left_curve_rad, right_curve_rad, center_offset


def make_pipeline(M, Minv, mtx, dist, low_pass):
    """
    @brief pipeline closure
    """
    def pipeline(img):
        """
        @brief pipeline takes RGB img returns RGB img with line
        """
        undistorted = cv2.undistort(img, mtx, dist, None, mtx)
        combined_binary, color_binary = apply_sobel_and_hls(undistorted)
        combined_binary = apply_mask(combined_binary)
        img_size = (undistorted.shape[1], undistorted.shape[0])
        binary_warped = cv2.warpPerspective(combined_binary, M, img_size, flags=cv2.INTER_LINEAR)
        lane_img, left_curve_rad, right_curve_rad, center_offset = window_search(binary_warped, low_pass)
        unwarped = cv2.warpPerspective(lane_img, Minv, img_size, flags=cv2.INTER_LINEAR)
        unwarped = cv2.addWeighted(undistorted, 1., unwarped, 0.3, 0)
        cv2.putText(unwarped, 'Offset: {:.2f} m'.format(center_offset), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1., (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(unwarped, 'L rad: {:.2f} km'.format(left_curve_rad / 1000), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1., (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(unwarped, 'R rad: {:.2f} km'.format(right_curve_rad / 1000), (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1., (255, 0, 0), 2, cv2.LINE_AA)
        return unwarped

    return pipeline

