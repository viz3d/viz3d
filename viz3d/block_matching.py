import numpy as np
from skimage.util.shape import view_as_windows
from sklearn.metrics import mean_squared_error
from os.path import join
import cv2 as cv
import matplotlib.pyplot as plt
import logging
from keras.models import load_model
from viz3d.cnn.kitti_data import preprocess_image


# Create logger
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s : %(name)s : %(message)s", level=logging.INFO)


NO_DISPARITY_VALUE = 255


class BlockMatcher:
    def __init__(self, disparity_calculator, window_size):
        assert window_size % 2 == 1
        self.disparity_calculator = disparity_calculator
        self.window_size = window_size

    def calculate_disparity_map(self, image_left, image_right):
        # Check preconditions
        assert image_left.shape == image_right.shape

        # Calculate windows borders
        image_width = image_left.shape[1]
        image_height = image_left.shape[0]
        windows_x_count = image_width - self.window_size + 1
        windows_y_count = image_height - self.window_size + 1

        # Prepare disparity calculation
        self.disparity_calculator.prepare(image_left, image_right, self.window_size)

        # Create empty disparity map
        disparity_map = np.zeros([windows_y_count, windows_x_count], dtype=np.int32)

        # Iterate all windows of the left image
        for y in range(windows_y_count):
            for x in range(windows_x_count):

                # Calculate disparities
                disparity = self.disparity_calculator.calculate_disparity(y, x)
                disparity_map[y, x] = disparity

            # Log progress
            logger.info("Processed %i of %i rows." % (y + 1, windows_y_count))

        # Pad image to make it the same shape as the original images
        padding = int(self.window_size / 2)
        disparity_map = np.pad(disparity_map, padding, "constant", constant_values=0)

        return disparity_map


class DisparityCalculator:

    def prepare(self, image_left, image_right, window_size):
        """
        Prepares calculation of the costs for a single image pair
        :param image_left:
        :param image_right:
        :param window_size:
        """
        raise NotImplementedError()

    def calculate_disparity(self, y_window, left_x_window):
        """
        Calculates the disparity for this window
        :param y_window: y coordinate of the window
        :param left_x_window: x coordinate of the window, only valid for the left windows
        :return: the calculated disparity or NO_DISPARITY_VALUE if no match was found
        """
        raise NotImplementedError()


class DisparityWindowed(DisparityCalculator):

    def __init__(self, max_disparity):
        self.window_size = None
        self.windows_left = None
        self.windows_right = None
        self.max_disparity = max_disparity

    def prepare(self, image_left, image_right, window_size):
        window_shape = (window_size, window_size)
        self.window_size = window_size
        self.windows_left = view_as_windows(image_left, window_shape)
        self.windows_right = view_as_windows(image_right, window_shape)

    def calculate_disparity(self, y_window, left_x_window):
        # Get right window row
        # Calculate row extending max_disparity to the left and right, but not out of bounds
        row_start = max(0, left_x_window - self.max_disparity)  # Do not extend to negative values
        row_end = min(self.windows_right.shape[1], left_x_window + self.max_disparity)  # Do not extend over the windows count
        # Get row
        windows_right_row = self.windows_right[y_window, row_start:row_end]

        # Get left window and duplicate it in a row
        window_left = self.windows_left[y_window, left_x_window]
        windows_row_count = windows_right_row.shape[0]
        windows_left_row = np.zeros([windows_row_count, self.window_size, self.window_size], dtype=window_left.dtype)
        windows_left_row[...] = window_left

        # Call subclass' cost
        costs = self.cost_windowed(windows_left_row, windows_right_row)

        # Get best match
        match = np.argmin(costs)
        if costs[match] == np.finfo(costs.dtype).max:
            match = -1

        # Calculate disparity
        disparity = NO_DISPARITY_VALUE
        if match != -1:
            right_x = row_start + match
            disparity = left_x_window - right_x

        if disparity < 0:
            disparity = NO_DISPARITY_VALUE

        return disparity

    def cost_windowed(self, windows_left_row, windows_right_row):
        raise NotImplementedError("Has to be implemented in subclass!")


class DisparityMse(DisparityWindowed):

    def __init__(self, max_disparity, threshold):
        super().__init__(max_disparity)
        self.threshold = threshold

    def cost_windowed(self, windows_left_row, windows_right_row):
        mse = ((windows_left_row - windows_right_row) ** 2).mean(axis=(1, 2))
        mse[mse > self.threshold] = np.finfo(mse.dtype).max
        return mse


class DisparityCnn(DisparityWindowed):

    def __init__(self, max_disparity, model_path):
        super().__init__(max_disparity)
        self.model = load_model(model_path)

    def cost_windowed(self, windows_left, windows_right):
        classes = self.model.predict([windows_left[..., np.newaxis], windows_right[..., np.newaxis]])
        costs = 1 - classes[:, 1]
        costs[classes[:, 0] > classes[:, 1]] = 1
        return costs


class DisparityCnnDot(DisparityCalculator):

    def __init__(self, model_path, max_disparity):
        self.model = load_model(model_path)
        self.max_disparity = max_disparity

    def prepare(self, image_left, image_right, window_size):
        # Apply model to both images to extract features
        logger.info("Starting to calculate features")
        # Prepare inputs
        image_left_batch = image_left[np.newaxis, ..., np.newaxis]
        image_right_batch = image_right[np.newaxis, ..., np.newaxis]
        # Calculate
        self.features_left = self.model.predict(image_left_batch)
        self.features_right = self.model.predict(image_right_batch)
        # Remove batch indices
        self.features_left = self.features_left[0]
        self.features_right = self.features_right[0]
        logger.info(" ... Done")

    def calculate_disparity(self, y_window, left_x_window):

        # Get feature row for right windows
        # Calculate row extending max_disparity to the left and right, but not out of bounds
        row_start = max(0, left_x_window - self.max_disparity)  # Do not extend to negative values
        row_end = left_x_window + 1
        # Get row
        features_right_row = self.features_right[y_window, row_start:row_end]

        # Get feature vector for left window
        features_left = self.features_left[y_window, left_x_window]
        features_left = features_left[np.newaxis, ...]

        # Calculate dot products for each combination of left feature vector and right feature vectors
        classification = np.sum(features_right_row * features_left, axis=-1)  # We skip softmax here, as it does not change the argmax

        # Calculate disparity
        match = np.argmax(classification)
        disparity = NO_DISPARITY_VALUE
        if match != -1:
            right_x = row_start + match
            disparity = left_x_window - right_x

        return disparity



def main():
    path = "data/data_stereo_flow/training/"
    image_name = "000160_10.png"
    image_left = cv.imread(join(path, "colored_0", image_name), cv.IMREAD_GRAYSCALE)
    image_right = cv.imread(join(path, "colored_1", image_name), cv.IMREAD_GRAYSCALE)
    disparity_map_correct = cv.imread(join(path, "disp_noc", image_name), cv.IMREAD_GRAYSCALE)

    # Preprocess images
    image_left = preprocess_image(image_left)
    image_right = preprocess_image(image_right)

    # Calculate disparity map
    #disparity_calculator = DisparityCnn("models/experimental_cnn/model.h5")
    #disparity_calculator = DisparityMse(100, 10000000)
    disparity_calculator = DisparityCnnDot("models/experimental_cnn_dot/model.h5", 100)
    block_matcher = BlockMatcher(disparity_calculator, 19)
    disparity_map = block_matcher.calculate_disparity_map(image_left, image_right)

    #disparity_map_correct_flat = disparity_map_correct[4:-4, 4:-4].reshape(-1)
    #costs_all_flat = costs_all.min(axis=2).reshape(-1)
    #plt.scatter(disparity_map_correct_flat, costs_all_flat, alpha=0.2)
    #plt.show()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    ax1.imshow(image_left, cmap="gray")
    ax3.imshow(image_right, cmap="gray")
    ax2.imshow(disparity_map, cmap="gray")
    ax4.imshow(disparity_map_correct, cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()
