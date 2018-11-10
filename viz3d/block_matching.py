import numpy as np
from skimage.util.shape import view_as_windows
from sklearn.metrics import mean_squared_error
from os.path import join
import cv2 as cv
import matplotlib.pyplot as plt
import logging
from keras.models import load_model


# Create logger
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s : %(name)s : %(message)s", level=logging.INFO)


class BlockMatcher:
    def __init__(self, cost_function, window_size):
        assert window_size % 2 == 1
        self.cost_function = cost_function
        self.window_size = window_size

    def calculate_disparity_map(self, image_left, image_right):
        # Check preconditions
        assert image_left.shape == image_right.shape

        # Sample windows from both images
        window_shape = (self.window_size, self.window_size)
        windows_left = view_as_windows(image_left, window_shape)
        windows_right = view_as_windows(image_right, window_shape)
        windows_x_count = windows_left.shape[1]
        windows_y_count = windows_left.shape[0]

        # Create empty disparity map
        disparity_map = np.zeros([windows_y_count, windows_x_count], dtype=np.int32)

        # DEBUG costs stats
        costs_all = np.zeros([windows_y_count, windows_x_count, windows_x_count], dtype=np.float64)

        # Iterate all windows of the left image
        for y in range(windows_y_count):
            for x in range(windows_x_count):

                # Retrieve relevant windows
                window_left = windows_left[y, x]
                windows_right_row = windows_right[y, :]

                # Create evaluation batch for epipolar line
                windows_left_row = self.create_row(window_left, windows_right_row)

                # Calculate costs
                costs = self.cost_function(windows_left_row, windows_right_row)
                costs_all[y, x] = costs

                # Get best match
                match = np.argmin(costs)
                if costs[match] == np.finfo(costs.dtype).max:
                    match = -1

                # Calculate disparity
                if match != -1:
                    disparity = x - match
                    disparity_map[y, x] = disparity

            # Log progress
            logger.info("Processed %i of %i rows." % (y + 1, windows_y_count))

        # Pad image to make it the same shape as the original images
        padding = int(self.window_size / 2)
        disparity_map = np.pad(disparity_map, padding, "constant", constant_values=0)

        return disparity_map, costs_all

    def create_row(self, window_left, windows_right_row):
        windows_row_count = windows_right_row.shape[0]
        windows_left_row = np.zeros([windows_row_count, self.window_size, self.window_size], dtype=window_left.dtype)
        windows_left_row[...] = window_left
        return windows_left_row


def cost_mse(windows_left, windows_right):
    mse = ((windows_left - windows_right) ** 2).mean(axis=(1, 2))
    mse[mse > 6] = np.finfo(mse.dtype).max
    return mse


def create_cost_cnn(path):
    model = load_model(path)

    def cost_cnn(windows_left, windows_right):
        classes = model.predict([windows_left[..., np.newaxis], windows_right[..., np.newaxis]])
        costs = 1 - classes[:, 1]
        costs[classes[:, 0] > classes[:, 1]] = 1
        return costs

    return cost_cnn


def main():
    path = "data/data_stereo_flow/training/"
    image_left = cv.imread(join(path, "colored_0", "000001_10.png"), cv.IMREAD_GRAYSCALE)
    image_right = cv.imread(join(path, "colored_1", "000001_10.png"), cv.IMREAD_GRAYSCALE)
    disparity_map_correct = cv.imread(join(path, "disp_noc", "000001_10.png"), cv.IMREAD_GRAYSCALE)

    block_matcher = BlockMatcher(create_cost_cnn("models/experimental_cnn/model.h5"), 9)
    disparity_map, costs_all = block_matcher.calculate_disparity_map(image_left, image_right)

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
