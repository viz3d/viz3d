import cv2 as cv
import numpy as np
import os
import os.path
import config
import logging
from camera_setup import SingleCameraSetup, StereoCameraSetup

# Load config
cfg = config.read_config()

# Create logger
logger = logging.getLogger(__name__)
logging.basicConfig(format=cfg["log"]["format"], level=cfg["log"]["level"])


def mark_calibration_points(image, image_marker=None, type="checkerboard"):
    """
    Draws the checkerboard corners on the image
    :param image: the image to process in grayscale
    :param image_marker: optional. Draws the markers on this image, can be the same as image
    :return: the corners found
    """

    # Mark calibration points
    dimension = cfg["calibration"][type]["dimension"]
    if type == "checkerboard":
        found, corners = cv.findChessboardCorners(image, dimension, None, flags=cv.CALIB_CB_ADAPTIVE_THRESH)
    else:
        found, corners = cv.findCirclesGrid(image, dimension, flags=cv.CALIB_CB_ASYMMETRIC_GRID)

    if found:
        if type == "checkerboard":
            # termination criteria
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.01)
            corners = cv.cornerSubPix(image, corners, (11, 11), (-1, -1), criteria)

        # Draw corners
        if image_marker is not None:
            cv.drawChessboardCorners(image_marker, dimension, corners, found)

        return found, corners
    return found, None


def get_points(image_lists, grid_type="checkerboard"):
    """
    Calculates the position of the corners from the checkerboard pattern
    :param left_images: list of the names from the left images
    :param right_images: list of the names from the right images
    :return: As numpy arrays: real world object points in millimetres,
    left image points in pixels, right image points in pixels
    """
    assert all(len(image_list) == len(image_lists[0]) for image_list in image_lists), \
        "Different file count for image sets!"
    image_count = len(image_lists[0])

    obj_points = [] # real world positions in millimetres
    point_lists = [] # image positions in pixels for each image list

    # Generate the general object points (same for every image, the checkerboard does not change)
    dimension = cfg["calibration"][grid_type]["dimension"]
    obj_point = np.zeros((dimension[0] * dimension[1], 3), dtype=np.float32)
    index = 0
    if grid_type == "checkerboard":
        for y in range(dimension[1]):
            for x in range(dimension[0]):
                obj_point[index][0] = x
                obj_point[index][1] = y
                index += 1
    elif grid_type == "circlesGrid":
        for y in range(dimension[1]):
            for x in range(dimension[0]):
                obj_point[index][0] = 2 * x + y % 2
                obj_point[index][1] = y
                index += 1
    obj_point = obj_point * cfg["calibration"][grid_type]["size"]

    # Calculate image points
    image_shape = None
    for i in range(image_count):
        # Read corresponding image from each image list as grayscale
        images = [cv.imread(image_list[i], cv.IMREAD_GRAYSCALE) for image_list in image_lists]

        # Store the shape of the image in reversed order (height x width -> width x height)
        image_shape = images[0].shape[::-1]

        # Mark the corner positions
        mark_results = [mark_calibration_points(image, type=grid_type) for image in images]

        # Only process if both images have a pattern found
        found_all = all(result[0] for result in mark_results)
        if found_all:
            obj_points.append(obj_point)
            corners = [result[1] for result in mark_results]
            point_lists.append(corners)

    # Convert to numpy arrays
    return np.asarray(obj_points, dtype=np.float32), \
           np.asarray(point_lists, dtype=np.float32), \
           image_shape


def calibrate_camera(obj_points, image_points, image_shape):
    """
    Calibrates a camera
    :param obj_points: The checkerboard corner positions on the checkerboard in millimetres
    :param image_points: The checkerboard corner positions in pixel for each image
    :param image_shape: The input image shape
    :return: The camera calibration matrix and the distortion coefficients as returned by cv.calibrateCamera
    """

    # Calibrate camera
    rms_error, camera_matrix, distortion_coeffs, r_vecs, t_vecs = \
        cv.calibrateCamera(obj_points, image_points, image_shape, None, None)

    # Calculate l2 reprojection error
    l2_error = 0
    num_images = obj_points.shape[0]
    for i in xrange(num_images):
        # Reproject points
        image_points_projected, _ = cv.projectPoints(obj_points[i], r_vecs[i], t_vecs[i], camera_matrix,
                                                     distortion_coeffs)
        # Calculate projection error
        num_points = len(image_points_projected)
        error = cv.norm(image_points[i], image_points_projected, cv.NORM_L2) / num_points
        l2_error += error
    l2_error /= num_images

    # Log
    logger.info("Calibrated camera. Reproject errors: rms=%f norm_l2=%f" % (rms_error, l2_error))

    setup = SingleCameraSetup(camera_matrix, distortion_coeffs, tuple(image_shape))
    return setup


def stereo_calibrate(left_image_points, right_image_points, setup_left, setup_right, image_shape, obj_points):

    # Stereo calibrate using left and right camera intrinsics (i.e. setups)
    ret, _, _, _, _, stereo_rotation, stereo_translation, stereo_essential, stereo_fundamental = \
        cv.stereoCalibrate(obj_points,
                           left_image_points,
                           right_image_points,
                           setup_left.camera_matrix,
                           setup_left.distortion_coeffs,
                           setup_right.camera_matrix,
                           setup_right.distortion_coeffs,
                           image_shape,
                           criteria=(cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 100, 1e-5),  # Optimization termination criteria
                           flags=cv.CALIB_FIX_ASPECT_RATIO | cv.CALIB_FIX_INTRINSIC)  # fix intrinsics because they were already provided (and given as setups)

    # Log
    logger.info("Calibrated stereo cameras. Reproject error: %f" % ret)

    setup = StereoCameraSetup(setup_left, setup_right, stereo_rotation, stereo_translation, stereo_essential, stereo_fundamental)
    return setup


def find_images(captures_folders, keys):
    ids = [name[:name.index("-")] for name in os.listdir(captures_folders) if name.endswith(keys[0] + ".tif")]
    ids.sort()
    logger.info("Found %d scenes for calibration.", len(ids))
    image_lists = [[os.path.join(captures_folders, id + "-" + key + ".tif") for id in ids] for key in keys]
    return image_lists


def main():
    # Find captures
    keys = ["color", "left", "right"]
    captures_folders = os.path.join(cfg["workingDir"], "captures-full")
    image_lists = find_images(captures_folders, keys)

    # Get points in real world and on the images
    logger.info("Loading image points")
    obj_points, images_points, image_shape = get_points(image_lists, grid_type="circlesGrid")

    # Calibration of the intrinsic values
    setup_single_list = []
    for index, key in enumerate(keys):
        logger.info("Calibrating camera %s" % key)
        image_points = images_points[:, index, ...]  # image points for current camera / key
        setup_single = calibrate_camera(obj_points, image_points, image_shape)
        setup_single_list.append(setup_single)

    # Calibration of the stereo pairs
    camera_left = keys.index("left")
    camera_right = keys.index("right")
    camera_color = keys.index("color")
    stereo_camera_calibration = stereo_calibrate(images_points[:, camera_left, ...],
                                                 images_points[:, camera_right, ...],
                                                 setup_single_list[camera_left],
                                                 setup_single_list[camera_right],
                                                 image_shape,
                                                 obj_points)
    stereo_depth_calibration = stereo_calibrate(images_points[:, camera_left, ...],
                                                images_points[:, camera_color, ...],
                                                 setup_single_list[camera_left],
                                                 setup_single_list[camera_color],
                                                image_shape,
                                                obj_points)

    # Store the data
    stereo_camera_calibration.save("calibration-stereo.json")
    stereo_depth_calibration.save("calibration-stereo-depth.json")
    logger.info("Calibration data stored.")

    # Example loading code for testing
    # loaded = StereoCameraSetup.load("calibration-stereo.json")
    # print(loaded)


if __name__ == "__main__":
    main()
