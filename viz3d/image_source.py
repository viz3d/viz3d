import cv2 as cv
from abc import abstractmethod
from openni import openni2
import numpy as np


class ImageSource:
    def __init__(self):
        pass

    @abstractmethod
    def get_frame(self):
        """

        :return: gray frame from this image source
        """
        raise NotImplementedError()

    # TODO close streams


class ImageSourceRgb(ImageSource):

    def __init__(self):
        ImageSource.__init__(self)

    def get_frame(self):
        rgb = self.get_frame_rgb()
        gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
        return gray

    @abstractmethod
    def get_frame_rgb(self):
        """

        :return: rgb image from this data source
        """
        raise NotImplementedError()


class ImageSourceCv(ImageSourceRgb):

    def __init__(self, camera_index, fps):
        ImageSourceRgb.__init__(self)
        self.video_capture = cv.VideoCapture(camera_index)
        self.video_capture.set(cv.CAP_PROP_FPS, fps)

    def get_frame_rgb(self):
        ret, image = self.video_capture.read()  # produces a BGR image
        if not ret:
            raise IOError("Could not read stream!")

        image = image[:, ::-1, ::-1]  # reverse x-axis, BGR to RGB

        return image


class ImageSourceNi(ImageSource):

    def __init__(self, stream, fps, image_size, pixel_format):
        ImageSource.__init__(self)
        self.stream = stream
        self.stream.set_video_mode(openni2.VideoMode(pixelFormat=pixel_format, fps=fps, resolutionX=image_size[0], resolutionY=image_size[1]))
        self.stream.start()

    def get_frame(self):
        frame = self.stream.read_frame()
        frame_np = np.frombuffer(frame.get_buffer_as_uint16(), np.uint16)
        frame_np = frame_np.reshape(frame.height, frame.width)
        return frame_np


class ImageSourceInfrared(ImageSourceNi):

    def __init__(self, device, fps, image_size, max_value=1024):
        ImageSourceNi.__init__(self, device.create_ir_stream(), fps, image_size, openni2.PIXEL_FORMAT_GRAY16)
        self.max_value = max_value

    def get_frame(self):
        frame = ImageSourceNi.get_frame(self)
        frame = frame.astype(np.float32) * (255.0 / 1024.0)
        frame = frame.astype(np.uint8)
        return frame


class ImageSourceDepth(ImageSourceNi):

    def __init__(self, device, fps, image_size, align_depth_to_color=False):
        ImageSourceNi.__init__(self, device.create_depth_stream(), fps, image_size, openni2.PIXEL_FORMAT_DEPTH_1_MM)

        if align_depth_to_color:
            # Align and sync output of depth to color
            device.set_depth_color_sync_enabled(True)
            device.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
