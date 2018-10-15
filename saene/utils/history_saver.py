""" Saves the history of the parameters of an autoencoder

Author: Tim Silhan
"""

import pickle
import sys
from os.path import dirname, realpath
from utils.copy_dict import copy_dict
sys.path.append(dirname(dirname(realpath(__file__))))
from neural_network_config import NeuralNetworkConfig

class HistorySaver:
    """ Saves the history of the parameters of an autoencoder

    If a *history_path* is provided the history is restored from the file
    specified in the path.

    Attributes:
        learning_rates: List of learning rates
        momentums: List of momentums
        num_steps: List of the number of steps
        batch_sizes: List of batch_sizes
        activations: List of activations
        dropout_rates: List of dropout rates
        fitnesses: List of fitnesses
    """
    def __init__(self, history_path=None):
        self.learning_rates = []
        self.momentums = []
        self.num_steps = []
        self.batch_sizes = []
        self.activations = []
        self.dropout_rates = []
        self.fitnesses = []

        if history_path is not None:
            self.deserialize(history_path)

    def add_config(self, config, fitness=None):
        """ Adds a config to the history

        Args:
            config: NeuralNetworkConfig
            fitness: Fitness achieved with the config
        """
        self.learning_rates.append(float(config.learning_rate))
        self.momentums.append(float(config.momentum))
        self.num_steps.append(int(config.num_steps))
        self.batch_sizes.append(int(config.batch_size))
        self.activations.append(copy_dict(config.activations))
        self.dropout_rates.append(copy_dict(config.dropout_rates))
        if fitness is not None:
            self.fitnesses.append(float(fitness))

    def serialize(self, path):
        """ Serialize this history and save it to a file

        Args:
            path: The path where the history should be saved
        """
        with open(path, "wb") as hist_file:
            pickle.dump(self, hist_file)

    @staticmethod
    def deserialize(path):
        """ Deserializes a history

        The history is deserialized from the pickle file at *path*

        Args:
            path: The path where the history should be saved

        Returns:
            A HistorySaver
        """
        with open(path, "rb") as hist_file:
            deser = pickle.load(hist_file)
            return deser

    def copy(self):
        """ Copy this history """
        copy_hist = HistorySaver()
        copy_hist.learning_rates = list(self.learning_rates)
        copy_hist.momentums = list(self.momentums)
        copy_hist.num_steps = list(self.num_steps)
        copy_hist.batch_sizes = list(self.batch_sizes)
        copy_hist.fitnesses = list(self.fitnesses)

        copy_activations = []
        for activation_dict in self.activations:
            copy_activations.append(copy_dict(activation_dict))
        copy_hist.activations = copy_activations

        copy_dropouts = []
        for dropout_dict in self.dropout_rates:
            copy_dropouts.append(copy_dict(dropout_dict))
        copy_hist.dropout_rates = copy_dropouts

        return copy_hist


    def get_config(self, gen):
        """ Returns the config that was used in the generation *gen*

        Args:
            gen: Generation of which to extract the config
        Returns:
            NeuralNetworkConfig of that generation
        """
        config = NeuralNetworkConfig()
        config.learning_rate = self.learning_rates[gen]
        config.momentum = self.momentums[gen]
        config.num_steps = self.num_steps[gen]
        config.batch_size = self.batch_sizes[gen]
        config.activations = self.activations[gen]
        config.dropout_rates = self.dropout_rates[gen]

        return config

    def get_last_config(self):
        """ Gets the last (most recent) configuration that was used """
        return self.get_config(len(self.learning_rates) - 1)

    def __str__(self):
        attributes = [entry for entry in self.__dict__ if not entry.startswith("__")]
        pairs = dict([(key, self.__getattribute__(key)) for key in attributes])

        representation = ""
        for key, val in pairs.items():
            representation += "{}: {}\n".format(str(key).ljust(15), val)

        return representation



if __name__ == "__main__":
    HS = HistorySaver()
    CONF = NeuralNetworkConfig()
    for _ in range(4):
        HS.add_config(CONF)
        CONF.mutate()
    CONF.append_layer()
    for _ in range(4):
        HS.add_config(CONF)
        CONF.mutate()
    HS.serialize("save.pickle")
    HS.deserialize("save.pickle")
