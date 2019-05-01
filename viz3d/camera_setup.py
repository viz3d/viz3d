import numpy as np
import cv2 as cv
import json
import pickle


class SingleCameraSetup:

    def __init__(self, camera_matrix, distortion_coeffs, image_size):
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = distortion_coeffs
        self.image_size = image_size

    def to_dict(self):
        return {
            "cameraMatrix": self.camera_matrix.tolist(),
            "distortionCoeffs": self.distortion_coeffs.tolist(),
            "imageSize": self.image_size
        }

    @staticmethod
    def from_dict(data):
        camera_matrix = np.array(data["cameraMatrix"], dtype=np.float64)
        distortion_coeffs = np.array(data["distortionCoeffs"], dtype=np.float64)
        image_size = tuple(data["imageSize"])
        return SingleCameraSetup(camera_matrix, distortion_coeffs, image_size)

    @staticmethod
    def load(file):
        with open(file, "r") as f:
            data = json.load(f)

        return SingleCameraSetup.from_dict(data)

    def save(self, file):
        data = self.to_dict()
        with open(file, "w") as f:
            json.dump(data, f, indent=4)

    def undistort(self, frame):
        cv.undistort(frame, self.camera_matrix, self.distortion_coeffs)


class StereoCameraSetup:

    def __init__(self, setup_left, setup_right, stereo_rotation, stereo_translation, stereo_essential, stereo_fundamental):
        self.setup_left = setup_left
        self.setup_right = setup_right

        # Stereo params
        self.stereo_rotation = stereo_rotation
        self.stereo_translation = stereo_translation
        self.stereo_essential = stereo_essential
        self.stereo_fundamental = stereo_fundamental

    def to_dict(self):
        return {
            "setupLeft": self.setup_left.to_dict(),
            "setupRight": self.setup_right.to_dict(),
            "stereoRotation": self.stereo_rotation.tolist(),
            "stereoTranslation": self.stereo_translation.tolist(),
            "stereoEssential": self.stereo_essential.tolist(),
            "stereoFundamental": self.stereo_fundamental.tolist()
        }

    @staticmethod
    def from_dict(data):
        setup_left = SingleCameraSetup.from_dict(data["setupLeft"])
        setup_right = SingleCameraSetup.from_dict(data["setupRight"])
        stereo_rotation = np.array(data["stereoRotation"], dtype=np.float64)
        stereo_translation = np.array(data["stereoTranslation"], dtype=np.float64)
        stereo_essential = np.array(data["stereoEssential"], dtype=np.float64)
        stereo_fundamental = np.array(data["stereoFundamental"], dtype=np.float64)
        return StereoCameraSetup(setup_left, setup_right, stereo_rotation, stereo_translation, stereo_essential, stereo_fundamental)

    @staticmethod
    def load(file):
        with open(file, "r") as f:
            data = json.load(f)
        return StereoCameraSetup.from_dict(data)

    def save(self, file):
        data = self.to_dict()
        with open(file, "w") as f:
            json.dump(data, f, indent=4)


