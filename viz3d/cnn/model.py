import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import logging
import kitti_data


# Create logger
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


def create_model(image_left, image_right):
    batch_size = image_left.shape[0]

    # Convolutional layer 1
    filters = 32
    kernel_size = [5, 5]
    conv1_left = tf.layers.conv2d(image_left, filters=filters, activation=tf.nn.relu, kernel_size=kernel_size, padding="valid", name="conv1")
    conv1_right = tf.layers.conv2d(image_right, filters=filters, activation=tf.nn.relu, kernel_size=kernel_size, padding="valid", reuse=True, name="conv1")

    # Dense layer 2
    conv1_flat_left = tf.reshape(conv1_left, [batch_size, -1], name="conv1_flat_left")
    conv1_flat_right = tf.reshape(conv1_right, [batch_size, -1], name="conv1_flat_right")
    dense2_left = tf.layers.dense(conv1_flat_left, 200, activation=tf.nn.relu, name="dense2")
    dense2_right = tf.layers.dense(conv1_flat_right, 200, activation=tf.nn.relu, name="dense2", reuse=True)

    # Dense layer 3
    dense3_left = tf.layers.dense(dense2_left, 200, activation=tf.nn.relu, name="dense3")
    dense3_right = tf.layers.dense(dense2_right, 200, activation=tf.nn.relu, name="dense3", reuse=True)

    # Concatenate left and right vector
    dense3 = tf.concat([dense3_left, dense3_right], 1)

    # Dense layer 4
    dense4 = tf.layers.dense(dense3, 300, activation=tf.nn.relu, name="dense4")

    # Dense layer 5
    dense5 = tf.layers.dense(dense4, 300, activation=tf.nn.relu, name="dense5")

    # Dense layer 6
    dense6 = tf.layers.dense(dense5, 300, activation=tf.nn.relu, name="dense6")

    # Dense layer 7
    dense7 = tf.layers.dense(dense6, 300, activation=tf.nn.relu, name="dense7")

    # Output layer
    #output = tf.layers.dense(dense7, 2, activation=tf.nn.relu, name="output")
    output_weights = tf.Variable(tf.truncated_normal([300, 2], stddev=0.1))
    output_bias = tf.Variable(tf.constant(0.1, shape=[2]))
    output = tf.matmul(dense7, output_weights) + output_bias

    return output


def train():
    window_size = 9
    batch_size = 128

    # Create placeholders
    image_left = tf.placeholder(tf.float32, shape=[batch_size, window_size, window_size, 1], name="image_left")
    image_right = tf.placeholder(tf.float32, shape=[batch_size, window_size, window_size, 1], name="image_right")
    target_class = tf.placeholder(tf.int32, shape=[batch_size], name="target_class")

    # Apply model
    output = create_model(image_left, image_right)

    # Calculate loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_class, logits=output, name="loss")

    # Define training step
    train_step = tf.train.AdamOptimizer().minimize(loss)
    learning_rate = tf.placeholder(tf.float32, shape=[])
    #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Load data
    samples_left = np.load("data/samples_left.npy")
    samples_right = np.load("data/samples_right.npy")
    samples_class = np.load("data/samples_class.npy")
    # Prepare data
    samples_left = samples_left[..., np.newaxis]
    samples_right = samples_right[..., np.newaxis]
    samples_class = samples_class.astype(np.int32)

    #tf.initialize_all_variables()

    # Create a session and start training
    with tf.Session() as session:
        # Init variables
        tf.global_variables_initializer().run()

        for epoch in range(1000000):
            # Sample a random batch
            batch_indices = np.random.randint(0, samples_left.shape[0], batch_size)
            #batch_indices = np.arange(batch_size)
            batch_left = samples_left[batch_indices]
            batch_right = samples_right[batch_indices]
            batch_class = samples_class[batch_indices]

            # Do training step
            _, loss_value = session.run([train_step, tf.reduce_mean(loss)], feed_dict={
                image_left: batch_left,
                image_right: batch_right,
                target_class: batch_class,
                learning_rate: .01
            })

            # Log progress
            print("Loss at epoch {}: {}".format(epoch, loss_value))




def main():
    train()


if __name__ == "__main__":
    main()