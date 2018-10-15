""" A single autoencoder is trained or restored

The trained autoencoder can be used for the baseline comparison with the ea.
Autoencoders that were trained with the ae can be restored here to
reconstruct images.

Author: Tim Silhan
"""

from tensorflow.examples.tutorials.mnist import input_data
from autoencoder import Autoencoder
from neural_network_config import NeuralNetworkConfig
from utils.history_saver import HistorySaver
from utils.time_me import time_me
from fitnesses import lcmc_fitness, mse_fitness
from datesets import Datasets

def main():
    """ Train or restore an autoencoder """
    data = input_data.read_data_sets("./data/mnist/", one_hot=True)
    # data = Datasets("./data/year_prediction_msd/YearPredictionMSD.txt", 453715, 5000)

    base_path = "checkpoints/"
    ae_name = "18ae545f-f797-4724-8f96-e197929578c4"

    # reconstruct_images(base_path + ae_name, data)
    print("Fitness: ", get_fitness(base_path + ae_name, data, lcmc_fitness))
    # print(train_autoencoder(data))
    # print_lineage(base_path + ae_name)


def reconstruct_images(path, data):
    """ Reconstruct images with an autoencoder

    Args:
        path: Path to the autoencoder without extension
        data: A fucntion that provides the input data for the network.
    """
    autoencoder, _ = restore_autoencoder(path)
    print(autoencoder.config)

    autoencoder.reconstruct_images(data)

def get_fitness(path, data, fitness):
    """ Calculates the LCMC Metric for the autoencoder

    Args:
        path: Path to the autoencoder without extension
        data: A fucntion that provides the input data for the network.
        fitness: a fitness function for the autoencoder (LCMC, QNX, MSE)

    Returns:
        LCMC fitness value
    """
    autoencoder, _ = restore_autoencoder(path)
    print(autoencoder.config)

    return fitness(autoencoder, data, test=True)

def print_lineage(path):
    """ Print the lineage of the hyperparameters of the autoencoder

    This can be used to extract the lineage information into a spreadsheet.

    Args:
        path: Path to the autoencoder without extension
    """
    _, history = restore_autoencoder(path)
    print(history.get_last_config())
    print(history)

    print("Encoder:")
    for dropout_dict in history.dropout_rates:
        for dropout_value in dropout_dict["encoder"]:
            print(dropout_value)

    print("\n\nDecoder:")
    for dropout_dict in history.dropout_rates:
        for dropout_value in dropout_dict["decoder"]:
            print(dropout_value)

@time_me
def train_autoencoder(data):
    """ Train an autoencoder

    Args:
        data: A fucntion that provides the input data for the network.

    Returns:
        LCMC and MSE metric for the autoencoder that has been trained.
    """

    # Setup
    layers = calculate_layer_sizes(784, 200, 0.5)
    # Generations is needed to correct the number of training steps to match
    # the number of steps used in the evolutionary algorithm
    generations = 10
    config = NeuralNetworkConfig()
    # Start with a mutated config to have some variation between runs
    config.mutate()
    config.num_steps *= generations
    autoencoder = Autoencoder(config, layers[:2])

    # Training
    autoencoder.train(data)
    for index, layer_size in enumerate(layers[2:]):
        autoencoder.append_layer(layer_size)
        autoencoder.train(data, restore_layers=index+1)

    # Evaluation
    autoencoder.save_history()
    print(autoencoder.config)
    print(autoencoder.save_path)
    # autoencoder.reconstruct_images(data)

    return lcmc_fitness(autoencoder, data, True), mse_fitness(autoencoder, data, True)

def calculate_layer_sizes(start, end, ratio):
    """ Calculates the sizes of the layers of the ae

    Args:
        start: Input size of the network.
        end: Maximum output size of the encoder.
        ratio: Ratio of the size of one layer to the next.

    Returns:
        A list of layer sizes
    """
    layers = [start]
    while start > end:
        start = int(start * ratio)
        layers.append(start)

    return layers


def restore_autoencoder(save_path):
    """ Reconstructs an autoencoder and its history
    Args:
        save_path: The full path to the autoencoder without extension

    Returns:
        The restored autoencoder and its corresponding history
    """
    hist_saver = HistorySaver.deserialize(save_path + ".pickle")
    config = hist_saver.get_last_config()
    # layers = calculate_layer_sizes(90, 25, 0.5)
    # layers = calculate_layer_sizes(784, 10, 0.234)
    # layers = calculate_layer_sizes(784, 196, 0.5)
    layers = calculate_layer_sizes(784, 10, 0.115)
    # layers = calculate_layer_sizes(784, 2, 0.052)
    autoencoder = Autoencoder(config, layers, restore_path=save_path + ".ckpt")

    return autoencoder, hist_saver

if __name__ == "__main__":
    main()
