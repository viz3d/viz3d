import cv2
import time
import os.path
import config
import logging

# Load config
cfg = config.read_config()

# Create logger
logger = logging.getLogger("image")
logging.basicConfig(format=cfg["log"]["format"], level=cfg["log"]["level"])


if __name__ == "__main__":

    # Left and right from camera view
    left = cv2.VideoCapture(cfg["cameras"]["left"])
    right = cv2.VideoCapture(cfg["cameras"]["right"])

    # Set camera fps from config
    left.set(cv2.CAP_PROP_FPS, cfg["cameras"]["fps"])
    right.set(cv2.CAP_PROP_FPS, cfg["cameras"]["fps"])

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
        grayLeft = cv2.cvtColor(frameLeft, cv2.COLOR_BGR2GRAY)
        grayRight = cv2.cvtColor(frameRight, cv2.COLOR_BGR2GRAY)
        # Show images
        cv2.imshow('frameLeft', grayLeft)
        cv2.imshow('frameRight', grayRight)
        # Check for key input. Close on 'q', capture image on 'c'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            epoch = int(time.time())
            ret1 = cv2.imwrite(os.path.join(cfg["workingDir"], "captures", "%i-left.png" % epoch), grayLeft)
            ret2 = cv2.imwrite(os.path.join(cfg["workingDir"], "captures", "%i-right.png" % epoch), grayRight)
            logger.info("Captured images. Left=%r. Right=%r" % (ret1, ret2))

    left.release()
    right.release()
