""" Configuration for the neural network and the training process

Author: Tim Silhan
"""

import copy
import random
import numpy as np
import tensorflow as tf

from utils.linear_activation import linear_activation
from utils.copy_dict import copy_dict

class NeuralNetworkConfig:
    """ Configuration for the neural network and the training process.

    Attributes:
        learnins_rate: Learning Rate
        momentum: Momentum
        num_steps: Number of steps for each training session
        batch_size: Number of samples for each training step
        activations: Dictionary of activation functions. Separated into
            encoder and decoder activation list.
        dropout_rates: Dictionary of dropout rates for training
    """

    def __init__(self, num_layers=1):
        """ Initializes the configuration
        Args:
            num_layers: The number of layers the configuration should be
                initialized for.
        """
        self.learning_rate = 0.05
        self.momentum = 0.1
        self.num_steps = 5000
        self.batch_size = 256
        self.activations = {"encoder": [], "decoder": []}
        self.dropout_rates = {"encoder": [], "decoder": []}

        for _ in range(num_layers):
            self.append_layer()

        self.mutate()

    def mutate(self):
        """ Mutate the configuration. """
        self.learning_rate = np.clip(
            self.learning_rate + np.random.normal(0, 0.025), 0.00001, 1.0)
        self.momentum = np.clip(self.momentum + np.random.normal(0, 0.05), 0.0, 1.0)

        enc_rates = self.dropout_rates["encoder"]
        enc_rates += np.random.normal(0, 0.05, len(enc_rates))
        enc_rates = np.clip(enc_rates, 0.01, 0.99)
        self.dropout_rates["encoder"] = enc_rates

        dec_rates = self.dropout_rates["decoder"]
        dec_rates += np.random.normal(0, 0.05, len(dec_rates))
        dec_rates = np.clip(dec_rates, 0.01, 0.99)
        self.dropout_rates["decoder"] = dec_rates

    def crossover(self, config):
        """ Crossover with another config and return new configuration """
        pass

    def copy(self):
        """ Copy of the configuration instance

        Returns:
            A NeuralNetworkConfig
        """
        new_conf = copy.copy(self)
        new_conf.activations = copy_dict(self.activations)
        new_conf.dropout_rates = copy_dict(self.dropout_rates)

        return new_conf

    def get_random_activation(self):
        """ Returns a random activation function

        Returns:
            An activation function
        """
        # The last option symbolises no activation (identity)
        # activations = [tf.nn.sigmoid, tf.nn.relu, tf.nn.tanh, linear_activation]
        activations = [tf.nn.sigmoid, tf.nn.relu, tf.nn.tanh]

        return random.choice(activations)

    def append_layer(self):
        """ Appends a layer to the configuration
        This means that new activation functions and dropout rates are created
        """
        self.activations["encoder"].append(self.get_random_activation())
        self.activations["decoder"].insert(0, self.get_random_activation())

        #fixed activation
        # self.activations["encoder"].append(tf.nn.sigmoid)
        # self.activations["decoder"].insert(0, tf.nn.sigmoid)

        self.dropout_rates["encoder"] = np.append(self.dropout_rates["encoder"],
                                                  np.clip(np.random.normal(0.1, 0.05), 0.01, 0.99))
        self.dropout_rates["decoder"] = np.insert(self.dropout_rates["decoder"], 0,
                                                  np.clip(np.random.normal(0.1, 0.05), 0.01, 0.99))

    def __str__(self):
        attributes = [entry for entry in self.__dict__ if not entry.startswith("__")]
        pairs = dict([(key, self.__getattribute__(key)) for key in attributes])

        representation = ""
        for key, val in pairs.items():
            representation += "{}: {}\n".format(str(key).ljust(15), val)

        return representation

if __name__ == "__main__":
    CONFIG = NeuralNetworkConfig()
    print(CONFIG.dropout_rates)
    CONFIG.append_layer()
    print(CONFIG.dropout_rates)
    CONFIG.append_layer()
    print(CONFIG.dropout_rates)
    CONFIG.mutate()
    print(CONFIG.dropout_rates)
    CONFIG.mutate()
    print(CONFIG.dropout_rates)
    COPIED = CONFIG.copy()
    print("Copied: ", COPIED.dropout_rates)
