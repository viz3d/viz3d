from openni import openni2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import viz3d.config as config
import logging
import time
import os

# Load config
cfg = config.read_config()

# Create logger
logger = logging.getLogger("calibration")
logging.basicConfig(format=cfg["log"]["format"], level=cfg["log"]["level"])


def main():

    openni2.initialize("/home/t/development/programs/OpenNI_2.3.0.55/Linux/OpenNI-Linux-x64-2.3.0.55/Redist/")

    dev = openni2.Device.open_any()
    print(dev.get_device_info())

    frame_options = {
        "resolutionX": 640,
        "resolutionY": 480,
        "fps": cfg["cameras"]["fps"]
    }

    ir_stream = dev.create_ir_stream()
    ir_stream.set_video_mode(openni2.VideoMode(pixelFormat=openni2.PIXEL_FORMAT_GRAY16, **frame_options))

    depth_stream = dev.create_depth_stream()
    depth_stream.set_video_mode(openni2.VideoMode(pixelFormat=openni2.PIXEL_FORMAT_DEPTH_1_MM, **frame_options))

    dev.set_depth_color_sync_enabled(True)
    dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)  # aligns output of color and depth

    ir_stream.start()
    depth_stream.start()

    # The following code does not seem to work, so color data is not part of the data OpenNI provides
    # color_stream = dev.create_color_stream()
    # color_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_GRAY16,
    #                                                **frame_options))
    # color_stream.start()

    color_stream = cv.VideoCapture(cfg["cameras"]["depthCameraColor"])  # parameter is the camera index. Find by try'n'error
    color_stream.set(cv.CAP_PROP_FPS, cfg["cameras"]["fps"])

    # Seems to have no effect?
    # depth_stream.set_mirroring_enabled(True)
    # ir_stream.set_mirroring_enabled(True)

    # Left and right from stereo camera view
    left_stream = cv.VideoCapture(cfg["cameras"]["left"])
    right_stream = cv.VideoCapture(cfg["cameras"]["right"])

    # Set camera fps from config
    left_stream.set(cv.CAP_PROP_FPS, cfg["cameras"]["fps"])
    right_stream.set(cv.CAP_PROP_FPS, cfg["cameras"]["fps"])

    logger.info("Streams initialized")

    while True:

        ir_frame = ir_stream.read_frame()
        ir_image = normalize_frame(ir_frame)
        cv.imshow("IR Stream", ir_image.astype(np.float32) / 1024)

        depth_frame = depth_stream.read_frame()
        depth_image = normalize_frame(depth_frame)
        cv.imshow("Depth Stream", depth_image.astype(np.float32) / (4 * 1000))

        ret, color_image = color_stream.read()  # produces a BGR image
        color_image = color_image[:, ::-1, :]  # reverse x-axis
        color_image = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)
        cv.imshow("Color Stream", color_image)

        ret, left_image = left_stream.read()
        left_image = left_image[:, ::-1, :]
        left_image = cv.cvtColor(left_image, cv.COLOR_BGR2GRAY)
        cv.imshow("Stereo Left Stream", left_image)

        ret, right_image = right_stream.read()
        right_image = right_image[:, ::-1, :]
        right_image = cv.cvtColor(right_image, cv.COLOR_BGR2GRAY)
        cv.imshow("Stereo Right Stream", right_image)

        # Check for key input. Close on 'q', plot images on 'p', capture images on 'c'
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            logger.info("Quitting...")
            break
        elif key == ord('p'):
            logger.info("Plotting images")
            fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex=True, sharey=True)
            ax1.imshow(ir_image, cmap="gray")
            ax4.imshow(depth_image, cmap="gray")
            ax2.imshow(color_image, cmap="gray")  # convert BGR to RGB
            ax3.imshow(left_image, cmap="gray")
            ax6.imshow(right_image, cmap="gray")
            plt.show()
        elif key == ord('c'):
            epoch = int(time.time() * 1000)
            folder = os.path.join(cfg["workingDir"], "captures-full")
            cv.imwrite(os.path.join(folder, "%d-ir.tif" % epoch), (ir_image.astype(np.float32) * (255 / 1024)).astype(np.uint8))
            cv.imwrite(os.path.join(folder, "%d-depth.tif" % epoch), depth_image)
            cv.imwrite(os.path.join(folder, "%d-color.tif" % epoch), color_image)
            cv.imwrite(os.path.join(folder, "%d-left.tif" % epoch), left_image)
            cv.imwrite(os.path.join(folder, "%d-right.tif" % epoch), right_image)
            logger.info("Took capture %d" % epoch)

    ir_stream.close()
    depth_stream.close()
    color_stream.release()
    left_stream.release()
    right_stream.release()
    dev.close()
    openni2.unload()
    logger.info("Released all resources.")


def normalize_frame(frame):
    frame_np = np.frombuffer(frame.get_buffer_as_uint16(), np.uint16)
    frame_np = frame_np.reshape(frame.height, frame.width)
    return frame_np


if __name__ == "__main__":
    main()

