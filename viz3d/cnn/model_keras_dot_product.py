import matplotlib.pyplot as plt
import numpy as np
import logging
from os.path import join
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, concatenate, Input, Activation, Dot
from keras.callbacks import EarlyStopping, TensorBoard
import keras.backend as K


# Create logger
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


def create_model(image_shape):

    # Single model (siamese network applied to both images)
    filters = 32
    kernel_size = [5, 5]
    model_single = Sequential()
    model_single.add(Conv2D(filters, kernel_size, name="conv1", activation="relu", input_shape=image_shape))
    model_single.add(Flatten())
    model_single.add(Dense(200, activation="relu"))
    model_single.add(Dense(64, activation="relu"))

    # Combine features from both single models
    image_left_input = Input(shape=image_shape, name="InputLeft")
    image_right_input = Input(shape=image_shape, name="InputRight")
    image_left_features = model_single(image_left_input)
    image_right_features = model_single(image_right_input)

    # Calculate dot product
    features_double = Dot(axes=1)([image_left_features, image_right_features])

    # Calculate sigmoid for binary classification
    features_double = Activation("sigmoid")(features_double)

    model_double = Model(inputs=[image_left_input, image_right_input], outputs=[features_double], name="MatchingModel")
    return model_double, model_single


def train():
    window_size = 9
    batch_size = 128
    image_shape = (window_size, window_size, 1)
    working_dir = join("models", "experimental_cnn_dot")

    # Apply model
    model_double, model_single = create_model(image_shape)

    model_double.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    samples_train_class, samples_train_left, samples_train_right = load_samples("train")
    samples_validation_class, samples_validation_left, samples_validation_right = load_samples("validation")

    callbacks = [
        EarlyStopping(monitor="val_acc", min_delta=0, patience=20, verbose=1, mode="auto"),
        TensorBoard(log_dir=join(working_dir, "log"), histogram_freq=0, write_graph=True, write_images=True, batch_size=batch_size)
    ]

    model_double.fit(x=[samples_train_left, samples_train_right], y=samples_train_class, batch_size=batch_size,
              epochs=int(1e6), verbose=0, callbacks=callbacks,
              validation_data=([samples_validation_left, samples_validation_right], samples_validation_class))

    model_single.save(join(working_dir, "model.h5"))


def extract_batch(batch_size, samples_left, samples_right, samples_class):
    # Sample a random batch
    batch_indices = np.random.randint(0, samples_left.shape[0], batch_size)
    # batch_indices = np.arange(batch_size)
    batch_left = samples_left[batch_indices]
    batch_right = samples_right[batch_indices]
    batch_class = samples_class[batch_indices]

    return batch_left, batch_right, batch_class


def load_samples(group, filename_pattern="data/samples_%s_%s.npy"):
    # Load data
    samples_left = np.load(filename_pattern % (group, "left"))
    samples_right = np.load(filename_pattern % (group, "right"))
    samples_class_index = np.load(filename_pattern % (group, "class"))
    # Prepare data
    samples_left = samples_left[..., np.newaxis]
    samples_right = samples_right[..., np.newaxis]

    return samples_class_index, samples_left, samples_right


def main():
    train()


if __name__ == "__main__":
    main()
