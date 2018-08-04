import cv2 as cv
import numpy as np
import time
import os
import os.path
import config
import logging

cfg = config.read_config()

def mark_checkerboard_corners(image):
    """
    Draws the checkerboard corners on the image
    :param image: the image to process in grayscale
    :return: the corners found
    """
    dimension = cfg["calibration"]["checkerboard"]["dimension"]
    found, corners = cv.findChessboardCorners(image, dimension, None)
    if found:
        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv.cornerSubPix(image, corners, (11, 11), (-1, -1), criteria)

        # Draw corners
        cv.drawChessboardCorners(image, dimension, corners, found)

        return found, corners
    return found, None

def get_points(left_images, right_images):
    assert len(left_images) == len(right_images), "Different file count for left and right images!"
    obj_points = []
    left_img_points = []
    right_img_points = []

    dimension = cfg["calibration"]["checkerboard"]["dimension"]
    obj_point = np.zeros(dimension[0] * dimension[1], 3)
    index = 0
    for y in range(dimension[1]):
        for x in range(dimension[0]):
            obj_point[index][0] = x
            obj_point[index][1] = y
    obj_point = obj_point * cfg["calibration"]["checkerboard"]["size"]

    for i in range(len(left_images)):
        left = cv.imread(left_images[i], cv.LOAD_IMAGE_GRAY)
        right = cv.imread(right_images[i], cv.LOAD_IMAGE_GRAY)

        left_found, left_corners = mark_checkerboard_corners(left)
        right_found, right_corners = mark_checkerboard_corners(right)

        if left_found and right_found:
            obj_points.append(obj_point)
            left_img_points.append(left_corners)
            right_img_points.append(right_corners)

    return np.asarray(obj_points, dtype=np.float32), \
        np.asarray(left_img_points, dtype=np.float32), \
        np.asarray(right_img_points, dtype=np.float32)


if __name__ == "__main__":

    captures = os.path.join(cfg["workingDir"], "captures")
    left_images = [os.path.join(captures, f) for f in os.listdir(captures) if f.startswith("left-")]
    right_images = [os.path.join(captures, f) for f in os.listdir(captures) if f.startswith("right-")]

