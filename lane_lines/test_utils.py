
import cv2
import helpers
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

def test_camera_calibration(path, mtx, dist):
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


def test_warping(path, M, mtx, dist):
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

            pr1 = (100, 720)
            pr2 = (1200, 720)
            pr3 = (550, 450)
            pr4 = (750, 450)
            cv2.line(undistorted, pr1, pr2, (0, 255, 255), 5)
            cv2.line(undistorted, pr3, pr4, (0, 255, 255), 5)
            cv2.line(undistorted, pr1, pr3, (0, 255, 255), 5)
            cv2.line(undistorted, pr2, pr4, (0, 255, 255), 5)

            undistorted = helpers.apply_mask(undistorted)

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


def test_sobel_HLS(path, M, mtx, dist):
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
            combined_binary, color_binary = helpers.apply_sobel_and_hls(rgb)
            combined_binary = helpers.apply_mask(combined_binary)
            img_size = (rgb.shape[1], rgb.shape[0])
            warped = cv2.warpPerspective(combined_binary, M, img_size, flags=cv2.INTER_LINEAR)

            f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 9))
            f.tight_layout()
            ax1.imshow(rgb, cmap='gray')
            ax1.set_title('Original undistorted image', fontsize=10)
            ax2.imshow(color_binary)
            ax2.set_title('Stacked thresholds', fontsize=10)
            ax3.imshow(combined_binary, cmap='gray')
            ax3.set_title('S channel and gradient threshold combined', fontsize=10)
            ax4.imshow(warped, cmap='gray')
            ax4.set_title('Warped image', fontsize=10)
            plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0.05)
            plt.show()


def test_initial_sliding_window(path, M, Minv, mtx, dist):
    """
    @brief test initial sliding window algorithm
    """
    if (os.path.isdir(path)):
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if not os.path.isfile(file_path):
                continue

            img = cv2.imread(file_path)
            undistorted = cv2.undistort(img, mtx, dist, None, mtx)
            rgb = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
            combined_binary, color_binary = helpers.apply_sobel_and_hls(rgb)
            img_size = (rgb.shape[1], rgb.shape[0])
            binary_warped = cv2.warpPerspective(combined_binary, M, img_size, flags=cv2.INTER_LINEAR)

            # Assuming you have created a warped binary image called "binary_warped"
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
                # Draw the windows on the visualization image
                cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
                cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
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

            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
            left_fitx = np.rint(left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2])
            right_fitx = np.rint(right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2])

            left_fit= np.dstack((left_fitx, ploty))
            left_fit = np.int32([left_fit])
            right_fit= np.dstack((right_fitx, ploty))
            right_fit = np.int32([right_fit])

            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
            cv2.polylines(out_img, left_fit, False, [255,255,0], 8)
            cv2.polylines(out_img, right_fit, False, [255,255,0], 8)

            #function to test
            lane_img, left_curve_rad, right_curve_rad, center_offset = helpers.window_search(binary_warped)

            unwarped = cv2.warpPerspective(lane_img, Minv, img_size, flags=cv2.INTER_LINEAR)
            unwarped = cv2.addWeighted(rgb, 1., unwarped, 0.3, 0)

            f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 9))
            f.tight_layout()
            ax1.imshow(rgb, cmap='gray')
            ax1.set_title('Original undistorted image', fontsize=10)
            ax2.imshow(binary_warped, cmap='gray')
            ax2.set_title('Warped image', fontsize=10)
            ax3.imshow(unwarped)
            ax3.set_title('Unwarped result image', fontsize=10)
            ax4.imshow(out_img)
            ax4.set_title('Result image', fontsize=10)

            plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0.05)
            plt.show()


def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output


def find_window_centroids(image, window_width, window_height, margin):

    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)

    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))

    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(image.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,image.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,image.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
        # Add what we found for that layer
        window_centroids.append((l_center,r_center))

    return window_centroids


def test_convolution_window(path, M, mtx, dist):
    """
    @brief test convolution window algorithm
    """
    if (os.path.isdir(path)):
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if not os.path.isfile(file_path):
                continue

            img = cv2.imread(file_path)
            undistorted = cv2.undistort(img, mtx, dist, None, mtx)
            rgb = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
            combined_binary, color_binary = helpers.apply_sobel_and_hls(rgb)
            img_size = (rgb.shape[1], rgb.shape[0])
            warped = cv2.warpPerspective(combined_binary, M, img_size, flags=cv2.INTER_LINEAR)

            # window settings
            window_width = 50 
            window_height = 80 # Break image into 9 vertical layers since image height is 720
            margin = 100 # How much to slide left and right for searching

            window_centroids = find_window_centroids(warped, window_width, window_height, margin)

            # If we found any window centers
            if len(window_centroids) > 0:

                # Points used to draw all the left and right windows
                l_points = np.zeros_like(warped)
                r_points = np.zeros_like(warped)

                # Go through each level and draw the windows
                for level in range(0,len(window_centroids)):
                    # Window_mask is a function to draw window areas
                    l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
                    r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
                    # Add graphic points from window mask here to total pixels found
                    l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
                    r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

                # Draw the results
                template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
                zero_channel = np.zeros_like(template) # create a zero color channel
                template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
                warpage= np.dstack((warped, warped, warped))*255 # making the original road pixels 3 color channels
                output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results

            # If no window centers found, just display orginal road image
            else:
                output = np.array(cv2.merge((warped,warped,warped)),np.uint8)

            # Display the final results
            plt.imshow(output)
            plt.title('window fitting results')
            plt.show()


def test_make_pipeline(path, M, Minv, mtx, dist):
    """
    @brief test of make pipeline function
    """
    pipeline = helpers.make_pipeline(M, Minv, mtx, dist)

    if (os.path.isdir(path)):
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if not os.path.isfile(file_path):
                continue

            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res_img = pipeline(img)

            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
            f.tight_layout()
            ax1.imshow(img)
            ax1.set_title('Original Image', fontsize=50)
            ax2.imshow(res_img)
            ax2.set_title('Result Image', fontsize=50)
            plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
            plt.show()

