import cv2 as cv
import time
import os.path
import config
import logging
import calibration
import numpy as np

# Load config
cfg = config.read_config()

# Create logger
logger = logging.getLogger("image")
logging.basicConfig(format=cfg["log"]["format"], level=cfg["log"]["level"])


if __name__ == "__main__":

    # Left and right from camera view
    left = cv.VideoCapture(cfg["cameras"]["left"])
    right = cv.VideoCapture(cfg["cameras"]["right"])

    # Set camera fps from config
    left.set(cv.CAP_PROP_FPS, cfg["cameras"]["fps"])
    right.set(cv.CAP_PROP_FPS, cfg["cameras"]["fps"])

    while True:
        # Get images from both cameras
        ret, frameLeft = left.read()
        if not ret:
            logger.error("Cannot load left camera!")
            break
        ret, frameRight = right.read()
        if not ret:
            logger.error("Cannot load right camera!")
            break

        # Convert to grayscale images
        grayLeft = cv.cvtColor(frameLeft, cv.COLOR_BGR2GRAY)
        grayRight = cv.cvtColor(frameRight, cv.COLOR_BGR2GRAY)

        imageMarkerLeft = np.copy(frameLeft)
        imageMarkerRight = np.copy(frameRight)

        # Mark checkerboard corners
        calibration.mark_calibration_points(grayLeft, imageMarker=imageMarkerLeft)
        calibration.mark_calibration_points(grayRight, imageMarker=imageMarkerRight)

        # Show images
        cv.imshow('frameLeft', imageMarkerLeft)
        cv.imshow('frameRight', imageMarkerRight)

        # Check for key input. Close on 'q', capture image on 'c'
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            epoch = int(time.time() * 1000)
            ret1 = cv.imwrite(os.path.join(cfg["workingDir"], "captures", "%i-left.png" % epoch), frameLeft)
            ret2 = cv.imwrite(os.path.join(cfg["workingDir"], "captures", "%i-right.png" % epoch), frameRight)
            logger.info("Captured images. Left=%r. Right=%r" % (ret1, ret2))

    left.release()
    right.release()
