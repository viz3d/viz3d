import cv2 as cv
import numpy as np
import logging
import open3d
import stereo
import calibration
import config


# Load config
cfg = config.read_config()

# Create logger
logger = logging.getLogger("point_cloud")
logging.basicConfig(format=cfg["log"]["format"], level=cfg["log"]["level"])


def main():
    # Load calibration data
    calibration_data = calibration.CalibrationData.load("calibration.json")

    # Calc rectification maps
    rectification_maps, disp_to_depth = stereo.get_undistorted_rectification_maps(calibration_data)

    # Load images
    left_image = cv.imread("captures/1533479723660-left.png")
    right_image = cv.imread("captures/1533479723660-right.png")

    # Convert to grayscale images
    left_gray = cv.cvtColor(left_image, cv.COLOR_BGR2GRAY)
    right_gray = cv.cvtColor(right_image, cv.COLOR_BGR2GRAY)

    # Rectify images
    left_rectified, right_rectified = stereo.rectify(left_gray, right_gray, rectification_maps)

    # Calculate disparity map
    stereoBm = cv.StereoBM_create(numDisparities=272, blockSize=5)
    disparity = stereoBm.compute(left_rectified, right_rectified)
    # Convert disparity map to float32 map. This is important for the 3d reprojection to work
    disparity = disparity.astype(np.float32) * 1.0/255

    # Calculate distances
    distances = cv.reprojectImageTo3D(disparity, disp_to_depth, True)
    # Calculate points. distances is of shape w*h*3 for a w*h image.
    # Convert it to a n*3 array (n=w*h), e.g.: an array of points
    points = np.reshape(distances, (-1, 3))

    # Get colors for each point from the left (unrectified) image of the camera.
    colors = np.reshape(left_image, (-1, 3)).astype(np.float32) / 255.0

    # Transform BGR to RGB
    colors = colors[..., ::-1]

    # Filter points. Remove all points with very low z or z bigger than a certain value.
    # The disparity values for these points were not calculated correctly
    points_mask = np.logical_and(points[..., 2] > -100000, points[..., 2] < -2600)
    colors = colors[points_mask]
    points = points[points_mask]

    # Render point cloud
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(points * np.array((1, 1, -1)))
    pcd.colors = open3d.Vector3dVector(colors)
    open3d.draw_geometries([pcd])


if __name__ == "__main__":
    main()