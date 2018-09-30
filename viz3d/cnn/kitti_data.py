import numpy as np
import cv2 as cv
import os
import random
import math
import logging
import matplotlib.pyplot as plt


# Create logger
logger = logging.getLogger("image")
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


def preprocess_image(image):
    image = image.astype(np.float32)
    mean = image.mean()
    std = image.std()
    image = (image - mean) / std
    return image


def load_image_pairs(dataset_dir):
    # Index files
    files = []
    for name in os.listdir(os.path.join(dataset_dir, "disp_noc")):
        files.append((
            os.path.join(dataset_dir, "image_0", name),
            os.path.join(dataset_dir, "image_1", name),
            os.path.join(dataset_dir, "disp_noc", name)
        ))

    # Retrieve samples
    samples = []
    for (file_left, file_right, file_disp) in files:
        # Load images
        image_left = cv.imread(file_left, cv.IMREAD_GRAYSCALE)
        image_right = cv.imread(file_right, cv.IMREAD_GRAYSCALE)
        image_disp = cv.imread(file_disp, cv.IMREAD_GRAYSCALE)
        assert image_left.shape == image_right.shape == image_disp.shape
        # Preprocess
        image_left = preprocess_image(image_left)
        image_right = preprocess_image(image_right)
        # Yield
        yield (image_left, image_right, image_disp)


def build_samples_from_image(image_pair, num_samples, window_size=9, n_low=4, n_high=8, p_high=1):
    assert window_size % 2 == 1

    # Calculate intervals for sample offsets
    offset_neg_set = np.concatenate((np.arange(-n_high, -n_low + 1, dtype=np.int), np.arange(n_low, n_high + 1, dtype=np.int)))
    offset_pos_set = np.arange(-p_high, p_high+1)

    # Get images
    image_left = image_pair[0]
    image_right = image_pair[1]
    image_disp = image_pair[2]

    # Create samples for image pair
    # Get sample bounds
    width = image_disp.shape[1]
    height = image_disp.shape[0]
    window_size_half = int(window_size / 2)
    start_x = window_size_half + n_high
    end_x = width - window_size_half - n_high
    start_y = window_size_half
    end_y = height - window_size_half
    # Create sample points
    x_values = np.arange(start_x, end_x)
    y_values = np.arange(start_y, end_y)
    xx, yy = np.meshgrid(x_values, y_values)
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    points = np.stack([xx, yy], axis=1)
    np.random.shuffle(points)

    # Create all samples
    sample_counter = 0
    point_counter = 0
    while sample_counter < num_samples:

        # Assert that there are still remaining points
        assert(point_counter < points.shape[0])

        # Get point for current points
        x, y = points[point_counter]
        point_counter += 1

        # Create sample
        disp = image_disp[y, x]
        if disp != 0:

            # Calculate offset positions
            offset_neg = offset_neg_set[np.random.randint(0, offset_neg_set.shape[0])]
            offset_pos = offset_pos_set[np.random.randint(0, offset_pos_set.shape[0])]
            x_neg = x - disp + offset_neg
            x_pos = x - disp + offset_pos

            # Continue to next point if right window is out of bounds
            if (not (start_x <= x_neg < end_x)) or (not (start_x <= x_pos < end_x)):
                continue

            # Create left patch
            patch_left = extract_patch(image_left, x, y, window_size_half)

            # Create right neg and pos patch
            patch_right_neg = extract_patch(image_right, x_neg, y, window_size_half)
            patch_right_pos = extract_patch(image_right, x_pos, y, window_size_half)

            yield (patch_left, patch_right_neg, 0)
            yield (patch_left, patch_right_pos, 1)

            sample_counter += 2


def extract_patch(image,  x, y, window_size_half):
    x_start = x - window_size_half
    x_end = x + window_size_half + 1
    y_start = y - window_size_half
    y_end = y + window_size_half + 1
    patch = image[y_start : y_end, x_start : x_end]
    return patch


def build_samples(image_pairs, num_samples=int(1e6), window_size=9, n_high=8, n_low=4, p_high=1):

    samples_per_image = math.ceil(num_samples / len(image_pairs))

    samples_left = np.zeros([num_samples, window_size, window_size], dtype=np.float32)
    samples_right = np.zeros([num_samples, window_size, window_size], dtype=np.float32)
    samples_class = np.zeros([num_samples], dtype=np.uint8)

    sample_index = 0
    stop = False
    for pair_index, image_pair in enumerate(image_pairs):

        # Extract samples
        for sample in build_samples_from_image(image_pair, samples_per_image, window_size=window_size, n_high=n_high, n_low=n_low, p_high=p_high):
            samples_left[sample_index] = sample[0]
            samples_right[sample_index] = sample[1]
            samples_class[sample_index] = sample[2]
            sample_index += 1

            # Break if num_samples is reached
            if sample_index >= num_samples:
                stop = True
                break

        # Log
        logger.info("At image pair %r, got %r samples" % (pair_index, sample_index))

        # Break if num_samples was reached
        if stop:
            break

    return samples_left, samples_right, samples_class


def plot_sample(sample_left, sample_right, sample_class):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    if sample_class == 0:
        fig.suptitle("neg {}".format(sample_class))
    else:
        fig.suptitle("pos {}".format(sample_class))
    ax1.imshow(sample_left, cmap="gray")
    ax2.imshow(sample_right, cmap="gray")
    plt.show()


def main():

    # Fix seeds
    random.seed(42)
    np.random.seed(42)

    # Load images & shuffle
    image_pairs = list(load_image_pairs("data/data_stereo_flow/training"))
    random.shuffle(image_pairs)

    samples_left, samples_right, samples_class = build_samples(image_pairs, num_samples=int(1e6)) #, window_size=45, n_high=40, n_low=20, p_high=5)
    np.save("data/samples_left", samples_left)
    np.save("data/samples_right", samples_right)
    np.save("data/samples_class", samples_class)


if __name__ == "__main__":
    main()
