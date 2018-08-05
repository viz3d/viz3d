import cv2 as cv
import numpy as np
import config
import logging
import calibration
import matplotlib.pyplot as plt

# Load config
cfg = config.read_config()

# Create logger
logger = logging.getLogger("stereo")
logging.basicConfig(format=cfg["log"]["format"], level=cfg["log"]["level"])


def get_undistorted_rectification_maps(calibration_data):
    """
    Creates the undistortion and rectification maps from the calibration data for the left and right camera.
    :param calibration_data: The calibration data from calibration.py
    :return: The left undistortion map, the left rectification map, the right undistortion map,
    the right rectification map, the disparity to depth matrix
    """
    # Stereo rectification
    left_rectification, right_rectification, left_projection, right_projection, disp_to_depth, _, _ = \
        cv.stereoRectify(calibration_data.left_camera_matrix,
                         calibration_data.left_distortion_coeffs,
                         calibration_data.right_camera_matrix,
                         calibration_data.right_distortion_coeffs,
                         calibration_data.image_size,
                         calibration_data.stereo_rotation,
                         calibration_data.stereo_translation,
                         flags=0)

    # Individiual undistortion and rectification
    left_undistortion_map, left_rectification_map = cv.initUndistortRectifyMap(
        calibration_data.left_camera_matrix,
        calibration_data.left_distortion_coeffs,
        left_rectification,
        left_projection,
        calibration_data.image_size,
        cv.CV_32F)
    right_undistortion_map, right_rectification_map = cv.initUndistortRectifyMap(
        calibration_data.right_camera_matrix,
        calibration_data.right_distortion_coeffs,
        right_rectification,
        right_projection,
        calibration_data.image_size,
        cv.CV_32F)

    return left_undistortion_map, left_rectification_map, right_undistortion_map, right_rectification_map, disp_to_depth


def main():
    calibration_data = calibration.CalibrationData.load("calibration.json")

    left_undistortion_map, left_rectification_map, right_undistortion_map, right_rectification_map, disp_to_depth = get_undistorted_rectification_maps(calibration_data)

    left_image = cv.imread("captures/1533457912010-left.png", cv.IMREAD_GRAYSCALE)
    right_image = cv.imread("captures/1533457912010-right.png", cv.IMREAD_GRAYSCALE)

    left_rectified = cv.remap(left_image, left_undistortion_map, left_rectification_map, cv.INTER_LINEAR)
    right_rectified = cv.remap(right_image, right_undistortion_map, right_rectification_map, cv.INTER_LINEAR)

    fig, subplots = plt.subplots(5, 5)
    for i in range(0, 5):
        numDisparities = 80 + i * 16 * 3
        for j in range(0, 5):
            blockSize = 5 + j *4
            stereo = cv.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)
            disparity = stereo.compute(left_rectified, right_rectified)
            subplots[i][j].imshow(disparity, cmap="gray")
            subplots[i][j].set_title("%d / %d" % (numDisparities, blockSize))
    plt.show()


if __name__ == "__main__":
    main()
