import cv2 as cv
import numpy as np
import json
import os
import os.path
import config
import logging

# Load config
cfg = config.read_config()

# Create logger
logger = logging.getLogger("calibration")
logging.basicConfig(format=cfg["log"]["format"], level=cfg["log"]["level"])


class CalibrationData:

    def __init__(self, left_camera_matrix, left_distortion_coeffs, right_camera_matrix, right_distortion_coeffs,
                 stereo_rotation, stereo_translation, stereo_essential, stereo_fundamental):
        self.left_camera_matrix = left_camera_matrix
        self.left_distortion_coeffs = left_distortion_coeffs
        self.right_camera_matrix = right_camera_matrix
        self.right_distortion_coeffs = right_distortion_coeffs
        self.stereo_rotation = stereo_rotation
        self.stereo_translation = stereo_translation
        self.stereo_essential = stereo_essential
        self.stereo_fundamental = stereo_fundamental

    def save(self, filename):
        """
        Save calibration data to file
        :param filename: the name of the file
        """
        with open(filename, "w") as f:
            data = {
                "leftCameraMatrix": self.left_camera_matrix.tolist(),
                "leftDistortionCoeffs": self.left_distortion_coeffs.tolist(),
                "rightCameraMatrix": self.right_camera_matrix.tolist(),
                "rightDistortionCoeffs": self.right_distortion_coeffs.tolist(),
                "stereoRotation": self.stereo_rotation.tolist(),
                "stereoTranslation": self.stereo_translation.tolist(),
                "stereoEssential": self.stereo_essential.tolist(),
                "stereoFundamental": self.stereo_fundamental.tolist()
            }
            json.dump(data, f, indent=4)

    @staticmethod
    def load(filename):
        """
        Constructs a CalibrationData object from file
        :param filename: the name of the JSON store
        :return: the CalibrationData object
        """
        with open(filename) as f:
            data = json.load(f)
            return CalibrationData(
                np.array(data["leftCameraMatrix"], dtype=np.float64),
                np.array(data["leftDistortionCoeffs"], dtype=np.float64),
                np.array(data["rightCameraMatrix"], dtype=np.float64),
                np.array(data["rightDistortionCoeffs"], dtype=np.float64),
                np.array(data["stereoRotation"], dtype=np.float64),
                np.array(data["stereoTranslation"], dtype=np.float64),
                np.array(data["stereoEssential"], dtype=np.float64),
                np.array(data["stereoFundamental"], dtype=np.float64)
            )


def mark_checkerboard_corners(image, imageMarker=None):
    """
    Draws the checkerboard corners on the image
    :param image: the image to process in grayscale
    :param imageMarker: optional. Draws the markers on this image, can be the same as image
    :return: the corners found
    """
    dimension = cfg["calibration"]["checkerboard"]["dimension"]
    found, corners = cv.findChessboardCorners(image, dimension, None)
    if found:
        # termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv.cornerSubPix(image, corners, (11, 11), (-1, -1), criteria)

        # Draw corners
        if imageMarker is not None:
            cv.drawChessboardCorners(imageMarker, dimension, corners, found)

        return found, corners
    return found, None


def get_points(left_images, right_images):
    """
    Calculates the position of the corners from the checkerboard pattern
    :param left_images: list of the names from the left images
    :param right_images: list of the names from the right images
    :return: As numpy arrays: real world object points in millimetres,
    left image points in pixels, right image points in pixels
    """
    assert len(left_images) == len(right_images), "Different file count for left and right images!"

    obj_points = [] # real world positions in millimetres
    left_img_points = [] # left image positions in pixels
    right_img_points = [] # right image positions in pixels

    # Generate the general object points (same for every image, the checkerboard does not change)
    dimension = cfg["calibration"]["checkerboard"]["dimension"]
    obj_point = np.zeros((dimension[0] * dimension[1], 3), dtype=np.float32)
    index = 0
    for y in range(dimension[1]):
        for x in range(dimension[0]):
            obj_point[index][0] = x
            obj_point[index][1] = y
            index += 1
    obj_point = obj_point * cfg["calibration"]["checkerboard"]["size"]

    # Calculate image points
    image_shape = None
    for i in range(len(left_images)):
        # Read images as grayscale
        left = cv.imread(left_images[i], cv.IMREAD_GRAYSCALE)
        right = cv.imread(right_images[i], cv.IMREAD_GRAYSCALE)

        # Store the shape of the image in reversed order (height x width -> width x height)
        image_shape = left.shape[::-1]

        # Mark the corner positions
        left_found, left_corners = mark_checkerboard_corners(left)
        right_found, right_corners = mark_checkerboard_corners(right)

        # Only process if both images have a pattern found
        if left_found and right_found:
            obj_points.append(obj_point)
            left_img_points.append(left_corners)
            right_img_points.append(right_corners)

    # Convert to numpy arrays
    return np.asarray(obj_points, dtype=np.float32), \
           np.asarray(left_img_points, dtype=np.float32), \
           np.asarray(right_img_points, dtype=np.float32), \
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
    return camera_matrix, distortion_coeffs


def main():
    # Find captures
    captures = os.path.join(cfg["workingDir"], "captures")
    left_images = [os.path.join(captures, f) for f in os.listdir(captures) if f.endswith("-left.png")]
    left_images.sort()
    right_images = [os.path.join(captures, f) for f in os.listdir(captures) if f.endswith("-right.png")]
    right_images.sort()

    # Get points in real world and on the images
    obj_points, left_image_points, right_image_points, image_shape = get_points(left_images, right_images)

    # Calibration of the intrinsic values
    logger.info("Calibrating left camera")
    left_camera_matrix, left_distortion_coeffs = calibrate_camera(obj_points, left_image_points, image_shape)
    logger.info("Calibrating right camera")
    right_camera_matrix, right_distortion_coeffs = calibrate_camera(obj_points, right_image_points, image_shape)

    # Calibration of the stereo values
    logger.info("Calibrating stereo")
    ret, _, _, _, _, stereo_rotation, stereo_translation, stereo_essential, stereo_fundamental = \
        cv.stereoCalibrate(obj_points, left_image_points, right_image_points, left_camera_matrix,
                           left_distortion_coeffs, right_camera_matrix, right_distortion_coeffs, image_shape)
    logger.info("Calibrated camera. Reproject error: %f" % ret)

    # Store the data
    calibration_data = CalibrationData(left_camera_matrix, left_distortion_coeffs, right_camera_matrix, 
                                       right_distortion_coeffs, stereo_rotation, stereo_translation, 
                                       stereo_essential, stereo_fundamental)
    filename = "calibration.json"
    calibration_data.save(filename)
    logger.info("Calibration data stored to %s" % filename)

    # Example loading code for testing
    # loaded = CalibrationData.load("calibration.json")
    # print(loaded)


if __name__ == "__main__":
    main()
