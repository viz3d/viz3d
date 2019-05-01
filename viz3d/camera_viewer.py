from openni import openni2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import viz3d.config as config
import logging
import time
import os
import viz3d.calibration as calibration
import sys
import viz3d.stereo as stereo
from image_source import ImageSourceInfrared, ImageSourceDepth, ImageSourceCv
from camera_setup import StereoCameraSetup, SingleCameraSetup
import time

# Load config
cfg = config.read_config()

# Create logger
logger = logging.getLogger("calibration")
logging.basicConfig(format=cfg["log"]["format"], level=cfg["log"]["level"])


def open_astra_device(cfg, image_size):

    openni2.initialize(cfg["openniRedist"])

    dev = openni2.Device.open_any()
    print(dev.get_device_info())

    fps = cfg["cameras"]["fps"]
    ir_source = ImageSourceInfrared(dev, fps, image_size)
    depth_source = ImageSourceDepth(dev, fps, image_size, align_depth_to_color=True)
    color_source = ImageSourceCv(cfg["cameras"]["depthCameraColor"], fps)

    return ir_source, depth_source, color_source


def main():
    show_rectified = len(sys.argv) == 2 and sys.argv[1] == "rectify"

    if show_rectified:
        setup_left_right = StereoCameraSetup.load("calibration-stereo.json")
        setup_left_depth = StereoCameraSetup.load("calibration-stereo-depth.json")
        setup_left = setup_left_right.setup_left
        setup_right = setup_left_right.setup_right
        setup_depth = setup_left_depth.setup_right

    fps = cfg["cameras"]["fps"]
    image_size = (640, 480)

    # Seems to have no effect?
    # depth_stream.set_mirroring_enabled(True)
    # ir_stream.set_mirroring_enabled(True)

    # Left and right from stereo camera view
    left_source = ImageSourceCv(cfg["cameras"]["left"], fps)
    right_source = ImageSourceCv(cfg["cameras"]["right"], fps)

    ir_source, depth_source, color_source = open_astra_device(cfg, image_size)

    logger.info("Streams initialized")

    # Ensure captures folder exists
    folder = os.path.join(cfg["workingDir"], "captures-full")
    if not os.path.isdir(folder):
        logger.info("Created captures-full folder")
        os.mkdir(folder)

    while True:

        ir_frame = ir_source.get_frame()
        depth_frame = depth_source.get_frame()
        color_frame = color_source.get_frame()
        left_frame = left_source.get_frame()
        right_frame = right_source.get_frame()

        if show_rectified:
            left_frame = setup_left.undistort(left_frame)
            right_frame = setup_right.undistort(right_frame)
            color_frame = setup_depth.undistort(color_frame)

        depth_frame_scaled = depth_frame.astype(np.float32) / (4.0 * 1000)

        marker_type = "checkerboard"
        show_frame(ir_frame, "IR", marker_type=marker_type)
        show_frame(depth_frame_scaled, "Depth")
        show_frame(color_frame, "Color", marker_type=marker_type)
        show_frame(left_frame, "Left", marker_type=marker_type)
        show_frame(right_frame, "Right", marker_type=marker_type)

        # Check for key input. Close on 'q', plot images on 'p', capture images on 'c'
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            logger.info("Quitting...")
            break
        elif key == ord('p'):
            logger.info("Plotting images")
            fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex=True, sharey=True)
            ax1.imshow(ir_frame, cmap="gray")
            ax4.imshow(depth_frame, cmap="gray")
            ax2.imshow(color_frame, cmap="gray")
            ax3.imshow(left_frame, cmap="gray")
            ax6.imshow(right_frame, cmap="gray")
            plt.show()
        elif key == ord('c'):
            epoch = int(time.time() * 1000)
            cv.imwrite(os.path.join(folder, "%d-ir.tif" % epoch), ir_frame)
            cv.imwrite(os.path.join(folder, "%d-depth.tif" % epoch), depth_frame)
            cv.imwrite(os.path.join(folder, "%d-color.tif" % epoch), color_frame)
            cv.imwrite(os.path.join(folder, "%d-left.tif" % epoch), left_frame)
            cv.imwrite(os.path.join(folder, "%d-right.tif" % epoch), right_frame)
            logger.info("Took capture %d" % epoch)

    #ir_stream.close()
    #depth_stream.close()
    #color_stream.release()
    #left_stream.release()
    #right_stream.release()
    #dev.close()
    #openni2.unload()
    #logger.info("Released all resources.")


def show_frame(frame, window_name, marker_type=None):
    if marker_type is not None:
        frame_marked = frame.copy()
        calibration.mark_calibration_points(frame, image_marker=frame_marked, type=marker_type)
        frame = frame_marked
    cv.imshow(window_name, frame)


if __name__ == "__main__":
    main()
