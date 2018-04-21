
# import cv2
from moviepy.editor import VideoFileClip
import helpers
# import test_utils


def run():
    """
    @brief main entry point uses pipeline to process given video
    """
    low_pass = helpers.LowPassFilter()

    ret, mtx, dist = helpers.calibrateCamera('./../camera_cal/')
    # test_utils.test_camera_calibration('./../camera_cal/', mtx, dist)
    M, Minv = helpers.warp_matrix()
    # test_utils.test_warping('./../test_images', M, mtx, dist)
    # test_utils.test_color_spaces('./../test_images', M, mtx, dist)
    # test_utils.test_initial_sliding_window('./../test_images',
    #                                       M, Minv, mtx, dist)
    # test_utils.test_convolution_window('./../test_images', M, mtx, dist)
    # test_utils.test_make_pipeline('./../test_images', M, Minv, mtx, dist)

    pipeline = helpers.make_pipeline(M, Minv, mtx, dist, low_pass)

    output_file = './../output_project_video.mp4'
    clip1 = VideoFileClip('./../project_video.mp4')
    # clip1.save_frame('./39.0.png', 39.0)
    # clip1 = VideoFileClip('./../project_video.mp4').subclip(38,42)
    output_clip = clip1.fl_image(pipeline)
    output_clip.write_videofile(output_file, audio=False)


if __name__ == '__main__':
    run()
