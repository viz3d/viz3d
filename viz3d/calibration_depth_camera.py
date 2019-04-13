import cv2 as cv
import numpy as np
import viz3d.config as config
import logging
import os

# Load config
cfg = config.read_config()

# Create logger
logger = logging.getLogger("calibration")
logging.basicConfig(format=cfg["log"]["format"], level=cfg["log"]["level"])


def main():
    folder = os.path.join(cfg["workingDir"], "captures-full")
    ids = [name[:name.index("-")] for name in os.listdir(folder) if name.endswith("-color.tif")]
    ids.sort()

    images = []
    for id in ids:
        pass  # TODO load images and calibrate


if __name__ == "__main__":
    main()
