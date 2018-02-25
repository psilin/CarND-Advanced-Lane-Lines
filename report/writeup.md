## Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[video1]: ./project_video.mp4 "Video"
[image1]: ./figs/Fig1_cam_calib.png
[image2]: ./figs/Fig2_cam_calib.png
[image3]: ./figs/Fig3_warp.png
[image4]: ./figs/Fig4_warp.png
[image5]: ./figs/Fig5_warp_masked.png
[image6]: ./figs/Fig6_Sobel_HLS.png
[image7]: ./figs/Fig7_Sobel_HLS.png
[image8]: ./figs/Fig8_Sl_win.png
[image9]: ./figs/Fig9_Sl_win.png
[image10]: ./figs/Fig10_pipe.png
[image11]: ./figs/Fig11_pipe.png


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---

### Writeup / README

#### 0. General discussion

Code is provided with writeup.md or available [here](https://github.com/psilin/CarND-Advanced-Lane-Lines/). To run it I used virtual invironment based on `python==3.5.2` with packages needed to be installed are in `requirements.txt` file. Project can be 
installed using following command:
```python
python setup.py install
```

Then `run_lane_lines` script should be executed from `lane_lines` directory as all paths to pictures and videos are relative to this directory. On the other hand, I just run following command from `lane_line` directory:
```python
python core.py
```
as it is a main entry point to the project. It produces `output_project_video.mp4` from `project_video.mp4` drawing lane lines on it.

Project code structure is as follows. `lane_lines/core.py` contains handling video and main pipeline execution per frame. `lane_lines/helpers.py` contains all methods that are used in pipeline and pipeline function itself. `lane_lines/test_utils.py` contains 
various test, almost each functions from `helpers.py` is tested there. All pictures it this reported were produced by these tests.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

As has been already mentioned camera calibaration function contains in `helpers.py` file and is called `calibrateCamera()`. I took a standard approach from lesson. Converted image to grayscale. Then called `cv2.findChessboardCorners()` on it and obtained 
image points. Object points are well known as it is a plain `9x6` grid. Computed those 2 subsets for all images in `camera_cal` directory then concatenated all image points and all object points. 2 resulting sets were used in `cv2.calibrateCamera()` function
to obtain parameters needed for camera calibration. Below are 2 images on the left side of image is original chessboard image, on the right side is the same image but after calibration is done. It can be seen that calibration is done correctly.

![alt text][image1]

![alt text][image2]


### Pipeline (single images)

My pipeline function is implemented as a result of `make_pipeline()` closure function in `helpers.py` file so all undistort-related and warp-related parameters are stored in it.

#### 1. Provide an example of a distortion-corrected image.

First step is to call `cv2.undistort()` fuction using parameters obtained during camera calibration. Examples of undistorted images can be seen below as all of them are undistorted.

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Here I want to be slightly change the order in my pipeline and disucss perspective transform as it is a revatively standalone feature (although it is after Sobex-related and HLS-related function in my pipeline). I used the idea that a lane on straight 
segment of the road that looks like a trapeze to our camera should be changed to rectangle in beards-eye-view perspective. What I have done is described in `warp_matrix()` function in `helpers.py` file. I took a picture with staright lane (on picture below):

![alt text][image3]

I found coordinates of vertices of trapeze that represents segment of straight lane (painted in red on left side of picture above) and transformed in to rectangle (painted in black on right side of picture above). Vertices of trapeze were used as
source points and corresponding vertices of rectangle were used as destination points. Those to sets of point were used in as follows:

```python
    p1 = (315, 650)
    p2 = (1005, 650)
    p3 = (525, 500)
    p4 = (765, 500)

    src = np.float32([p1, p2, p3, p4])
    dst = np.float32([p1, p2, (p1[0], 360), (p2[0], 360)])

    img_size = (1280, 720)
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
```

This is source code of `warp_matrix()` fuction. I swapped source and destionation points to obtain inverse matrix to unwarp images with lane lines found. I hard coded that transformation under assumption that car perspective was not changing during video. 
On image below is an example of application warp transformation to curved part of lane:

![alt text][image4]

Here is the table containing source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 315, 650      | 315, 650      |
| 1005, 650     | 1005, 650     |
| 525, 500      | 315, 360      |
| 765, 500      | 1005, 360     |


#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. Code can be found in `apply_sobel_and_hls()` function in `helpers.py` file. I used standard approach from
lesson and the same thesholds. I applied SobelX operator using `cv2.Sobel()` function and S-channel of `HLS` image using `cv2.cvtColor()` function to transform image to `HLS` color scheme.
The result can be seen on image below:

![alt text][image6]

Left top image is original undistorted image. Top right image is image where SobelX is applied (green) and S-channel threshold (blue). Bottom left image is those to methods
applied combined (with using of mask-region that will be discussed near the end of writeup). Bottom right image is bottom left image after warping. Below another example of
the same images. But it is a bad case a shadows are really spoiling result (look at bottom right image):

![alt text][image7]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I implemented window search as it was in lesson. Code can be found in `helpers.py` file in `window_search()` funtion. I tried to use convolution-related method
instead of window search but as it was it performed much worse for me so I dropped it. Firstly, I applied histogram to bottom halw of image to obtain starting X coordinates
used in window search (left and right peaks on histogram). Then I divided Y-axis into 9 windows. I started from bottom of image and proceeded to its top. In `for` loop
I was counting all non-zero pixels that were inside of my window of search. If I realized that I had found more than 50 pixels I recentered starting X coordinate for
next window to mean value of previous window's pixels. All those windows of search are painted in green on image below (bottom left image). Non-zero pixels that fell into right windows are 
painted in red, non-zero pixels for left windows are painted in blue (bottom left image). Than I fitted a 2nd order polynome using all red pixels and another one using all blue pixels.
Curves based on those polynomes are painted in yellow (bottom left image). Lane that I had found was between those 2 curves. Unwrapped lane (that was between yellow curves) is shown
in geern on bottom right image.

![alt text][image8]

Below is the example of the case where this algorithm is not robust. Noize caused by shadows on lane resulted in 2 curves with curvature of different sign. That is clearly impossible in real world.

![alt text][image9]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

This code can be seen in `get_curvature()` method in `helpers.py` file. Again, I used it as it was in lesson. I found fitting polynomes that correspond to lane lines in pixel space.
Then I transformed points of this polynomes to real space using known constants and used this new points in polynome fitting function once again. That I used polynome coefficints and formula from 
lesson to calculate curvature in real space. Curvature is shown in red on right image below. Another one is offset of cat center from lane center. I found it in pixel space under assumption that
camera was mounted in car's center that I transformed offset to real space using constants above (code related to offset finding is last 10 lins of `window_search()` function).

![alt text][image11]

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Here is an example of my result on a test image. The radius of curvature is quite big as it is a straight segment of the road.

![alt text][image10]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Resulting video is in `output_project_video.mp4` file. Video is processed frame by frame using pipeline (`make_pipeline()` function). Code for video processing is in `core.py`. Pipeline consists of following steps:
 * undistortind image;
 * applying SobelX and HLS threshold to it to obtain binary image;
 * applying mask of interest to it (discussed later);
 * warping it to bird's eye view perspective;
 * finding lane lines with window search;
 * warping image back to car's view perspective using `Minv` matrix;
 * adding found lane lines to original undistorted image and drawing useful information on it.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Fistly, I implemented pipeline in plain frame by frame way. It was performing reasonably well through almost all of video but on both bridges where the shadows were there were big
outliers. Window search was reacting a little on black car as well (shown in curvature-related section 5). My first idea was that car is always on lane during the video, so I can apply
the region of interest. I set to 0 all pixels outside of trapeze and inside of triangle between the lane lines. Region of interest is shown on image below.

![alt text][image5]

Applying region of interest resulted in algorithm stopping to react on black car. Outliers on bridge became smaller. My next idea was to use smoothing as I could assume that lines were 
continuous, fitting polynome coefficients and curvature radious were continuous. I used simple low-pass 1st order filter `LowPassFilter` class in `helpers.py`. I applied it to fitting lane 
lines polynome coefficients so I could have some memory of what lane lines looked like on previous frames. It helped to decrease outliers values. I tried different values for `alpha` parameter 
and ended up with `alpha = 1./6.` as it greatly decreased outliers yet polynomes reacted quite good on curvature changing.

Though I did not manage to remove all outliers and lane lines were a little noisy. Possible steps are to make threshold tweaks in SobelX and HLS-related function as I just used arameters as it was.
Another idea is to use memory and filtering as much as possible as everything is quite continuous in this project (and no need to use window search from the very beginning on each frame). So memory 
of successful detections in non-noisy environment should be used. Another idea is to impelement outliers detection procedure. If everything is continuous obtained vaues should not differ much from 
filtered or remembered values. So if value differs much it should be rejected.

