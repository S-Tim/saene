"""
This module contains methods to load and prepare the MNIST data for the
use in a neural network

Author: Tim Silhan

Note:
    This module is currently not used because of more convenient alternatives.
"""

import tensorflow as tf
import numpy as np

def load_data(batch_size=100):
    """ Loads the MNIST data set using Keras

    The dataset is transformed from the 28*28 representation into a
    flat 784 (28*28) representation.

    Returns:
        Two tuples:
            - x_train, y_train: With shape (num_samples, 784) and (num_samples)
            - x_test, y_test: With shape (num_samples, 784) and (num_samples)
    """
    train, test = tf.keras.datasets.mnist.load_data()

    # Change shape from 28*28 to 784
    train = (np.reshape(train[0], (-1, 784)), train[1])
    test = (np.reshape(test[0], (-1, 784)), test[1])

    # Create datasets
    train = tf.data.Dataset.from_tensor_slices(train).shuffle(50000).repeat().batch(batch_size)
    test = tf.data.Dataset.from_tensor_slices(test).shuffle(10000).repeat().batch(batch_size)

    return train.make_one_shot_iterator(), test.make_one_shot_iterator()
