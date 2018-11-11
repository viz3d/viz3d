import logging
from os.path import join
import numpy as np
import keras.backend as K
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Conv2D, Input, Activation, BatchNormalization, RepeatVector, Reshape, Lambda
from keras.models import Sequential, Model


# Create logger
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s : %(name)s : %(message)s", level=logging.INFO)


def create_model_single(vector_size):

    input_shape = (None, None, 1)

    # Single model (conv network applied to each image patch)
    filters = 64
    kernel_size = [3, 3]
    batchnorm_epsilon = 1e-3
    model_single = Sequential()

    # Add input conv layer
    model_single.add(Conv2D(filters, kernel_size, input_shape=input_shape))
    model_single.add(BatchNormalization(epsilon=batchnorm_epsilon))
    model_single.add(Activation("relu"))

    # Add 7 conv layers with relu activation
    for i in range(7):
        model_single.add(Conv2D(filters, kernel_size))
        model_single.add(BatchNormalization(epsilon=batchnorm_epsilon))
        model_single.add(Activation("relu"))

    # Add output conv layer (linear activation to keep information from negative values)
    model_single.add(Conv2D(vector_size, kernel_size))
    model_single.add(BatchNormalization(epsilon=batchnorm_epsilon))

    return model_single


def create_model_training(model_single, window_size, max_disparity, vector_size):
    num_classes = max_disparity * 2 + 1

    # Apply single model on left image window
    input_left = Input(shape=(window_size, window_size, 1))
    features_left = model_single(input_left)
    features_left = Reshape((vector_size,))(features_left)  # Flatten features
    features_left = RepeatVector(num_classes)(features_left)  # Left window exists once, so repeat it for all classes

    # Apply single model on right image windows
    input_right = Input(shape=(window_size, window_size + max_disparity * 2, 1))
    features_right = model_single(input_right)
    features_right = Reshape((num_classes, vector_size))(features_right)  # Remove first dimension (y) as it is always 1

    # Calculate the dot product from left and right features
    # [(?, num_classes, vector_size), (?, num_classes, vector_size)] -> (?, num_classes)
    dot_combined = Lambda(lambda x: K.sum(x[0] * x[1], axis=-1, keepdims=False))([features_left, features_right])
    dot_combined = Activation("softmax")(dot_combined)  # Apply softmax smoothing

    # Create complete training model from the functional api
    model_training = Model(inputs=[input_left, input_right], outputs=[dot_combined])
    return model_training


def train():
    window_size = 19
    max_disparity = 100
    batch_size = 32
    working_dir = join("models", "experimental_cnn_dot")
    vector_size = 64  # Size of the vector representing each window

    # Create models
    # The single model is applied to each window_size*window_size image patch
    # The training model wraps the single model for training into our architecture
    model_single = create_model_single(vector_size)
    model_training = create_model_training(model_single, window_size, max_disparity, vector_size)

    # Compile training model
    model_training.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # Load training data
    filename_pattern = "data/samples_row_w%i_%%s_%%s.npy" % window_size
    num_classes = max_disparity * 2 + 1
    samples_train_class, samples_train_left, samples_train_right = load_samples("train", filename_pattern, num_classes)
    samples_validation_class, samples_validation_left, samples_validation_right = load_samples("validation", filename_pattern, num_classes)

    callbacks = [
        EarlyStopping(monitor="val_acc", min_delta=0, patience=20, verbose=1, mode="auto"),
        TensorBoard(log_dir=join(working_dir, "log"), histogram_freq=0, write_graph=True, write_images=True, batch_size=batch_size)
    ]

    # Fit model
    model_training.fit(x=[samples_train_left, samples_train_right],
                       y=samples_train_class,
                       batch_size=batch_size,
                       epochs=int(1e6),
                       verbose=0,
                       callbacks=callbacks,
                       validation_data=([samples_validation_left, samples_validation_right], samples_validation_class))

    # Store model
    model_single.save(join(working_dir, "model.h5"))


def load_samples(group, filename_pattern, num_classes):
    # Load data
    samples_left = np.load(filename_pattern % (group, "left"))
    samples_right = np.load(filename_pattern % (group, "right"))
    samples_class_index = np.load(filename_pattern % (group, "class"))
    # Prepare data
    samples_left = samples_left[..., np.newaxis]
    samples_right = samples_right[..., np.newaxis]
    # Three pixel error
    # Smooth the one hot encoding for similar disparities
    samples_class_onehot = np.zeros([samples_class_index.shape[0], num_classes])
    samples_indices = np.arange(samples_class_onehot.shape[0])
    samples_class_onehot[samples_indices, samples_class_index - 2] = 0.05
    samples_class_onehot[samples_indices, samples_class_index - 1] = 0.2
    samples_class_onehot[samples_indices, samples_class_index + 0] = 0.5
    samples_class_onehot[samples_indices, samples_class_index + 1] = 0.2
    samples_class_onehot[samples_indices, samples_class_index + 2] = 0.05

    return samples_class_onehot, samples_left, samples_right


def main():
    train()


if __name__ == "__main__":
    main()
