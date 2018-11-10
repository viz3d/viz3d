import matplotlib.pyplot as plt
import numpy as np
import logging
from os.path import join
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, concatenate, Input
from keras.callbacks import EarlyStopping, TensorBoard


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
    model_single.add(Dense(200, activation="relu"))

    # Combine features from both single models
    image_left_input = Input(shape=image_shape, name="InputLeft")
    image_right_input = Input(shape=image_shape, name="InputRight")
    image_left_features = model_single(image_left_input)
    image_right_features = model_single(image_right_input)
    double_features = concatenate(inputs=[image_left_features, image_right_features], name="ConcatFeatures")
    double_features = Dense(300, activation="relu")(double_features)
    double_features = Dense(300, activation="relu")(double_features)
    double_features = Dense(300, activation="relu")(double_features)
    double_features = Dense(300, activation="relu")(double_features)
    double_features = Dense(2, activation="softmax")(double_features)
    model_double = Model(inputs=[image_left_input, image_right_input], outputs=[double_features], name="MatchingModel")

    return model_double


def train():
    window_size = 9
    batch_size = 128
    image_shape = (window_size, window_size, 1)
    working_dir = join("models", "experimental_cnn")

    # Apply model
    model = create_model(image_shape)

    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    samples_train_class, samples_train_left, samples_train_right = load_samples("train")
    samples_validation_class, samples_validation_left, samples_validation_right = load_samples("validation")

    callbacks = [
        EarlyStopping(monitor="val_loss", min_delta=0, patience=10, verbose=0, mode="auto"),
        TensorBoard(log_dir=join(working_dir, "log"), histogram_freq=0, write_graph=True, write_images=True, batch_size=batch_size)
    ]

    model.fit(x=[samples_train_left, samples_train_right], y=samples_train_class, batch_size=batch_size,
              epochs=int(1e6), verbose=0, callbacks=callbacks,
              validation_data=([samples_validation_left, samples_validation_right], samples_validation_class))

    model.save(join(working_dir, "model.h5"))


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
    # One-hot encoding
    samples_class_onehot = np.zeros([samples_class_index.shape[0], 2])
    samples_class_onehot[np.arange(samples_class_onehot.shape[0]), samples_class_index] = 1

    return samples_class_onehot, samples_left, samples_right


def main():
    train()


if __name__ == "__main__":
    main()
