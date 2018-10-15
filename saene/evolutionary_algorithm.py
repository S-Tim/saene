""" Evolutionary algorithm to optimize stacked autoencoders

Autoencoders are evolved layerwise. Starting with (usually) one encoder and
decoder layer. The autoencoders are then trained and mutated over a number of
generations before appending the next layer. The evolutionary process stops
when the desired output size of the encoder is reached.

Author: Tim Silhan
"""

import sys
from tensorflow.examples.tutorials.mnist import input_data
from autoencoder import Autoencoder
from neural_network_config import NeuralNetworkConfig
from utils.time_me import time_me
from fitnesses import lcmc_fitness, mse_fitness
from ea_config import EAConfig
from datesets import Datasets

def main():
    """ Main function of the evolutionary algorithm """

    data = input_data.read_data_sets("./data/mnist/", one_hot=True, validation_size=5000)
    # data = Datasets("./data/year_prediction_msd/YearPredictionMSD.txt", 453715, 5000)

    best_individual = evolve(data)
    best_individual[0].save_history()
    print(best_individual[0].save_path)

    if len(sys.argv) > 1 and sys.argv[1] == "colab":
        print("Running on colab. Not reconstructing images.")
    else:
        best_individual[0].reconstruct_images(data)

@time_me
def evolve(data):
    """ Runs an evolutionary algorithm with the provided training data.

    Args:
        data: A fucntion that provides the input data for the network.

    Returns:
        (autoencoder, fitness) with best fitness value after evolution.
    """

    ea_config = EAConfig()

    # Create initial population
    population = initialize(ea_config.pop_size, ea_config.layer_size, ea_config.layer_ratio)

    current_layer = 0

    while ea_config.layer_size > ea_config.target_size:
        print("----------")
        print("Layer {}".format(current_layer + 1))
        print("----------")

        ea_config.layer_size = int(ea_config.layer_size * ea_config.layer_ratio)
        # Always add a new layer except after initialization
        # Freeze all previous layers
        if current_layer != 0:
            for auto_enc in population:
                # auto_enc[0].freeze_all_layers()
                auto_enc[0].append_layer(ea_config.layer_size)

        for gen in range(ea_config.gens_per_layer):
            # Rate population
            for i, auto_enc in enumerate(population):
                # Train the autoencoder
                training_score = auto_enc[0].train(data, restore_layers=current_layer)

                # Evaluate fitness of the autoencoder
                # Use negative lcmc because higher lcmc is better and the
                # selection starts with the lowest
                population[i] = (auto_enc[0], -lcmc_fitness(auto_enc[0], data))
                # MSE
                # population[i] = (auto_enc[0], mse_fitness(auto_enc[0], data))
                # print(training_score, population[i][1])


            # Always restore layers except with newly initialized layers
            if gen == 0:
                current_layer += 1

            # Select best subjects and discard others
            population.sort(key=lambda t: t[1])
            population = population[:int(ea_config.pop_size * ea_config.selection_ratio)]

            # Add configuration of the autoencoders to their history
            for auto_enc, fitness in population:
                auto_enc.history_saver.add_config(auto_enc.config, fitness)

            # Refill population
            children = []
            for i in range(ea_config.pop_size - len(population)):
                # (Almost) equal chances of offspring for all individuals
                children.append(create_child(population[i % len(population)]))
            population += children

            print("-" * 70)
            print("Best individual in generation {} has a loss of: {}".format(
                gen + 1, population[0][1]))
            print(population[0][0].config)
            print("-" * 70)

        print("Best individual in layer {} has a loss of: {}".format(
            current_layer, population[0][1]))
        print(population[0][0].config)

    return population[0]

def initialize(pop_size, layer_size, layer_ratio):
    """ Initializes the population to pop_size

    Args:
        pop_size: Individuals per generation
        layer_size: Size of the currently smallest layer
        layer_ratio: Ratio of new and old layer

    Returns:
        A list of (autoencoder, fitness) tuples of the newly initialized
        population.
    """

    next_layer_size = int(layer_size * layer_ratio)
    population = []

    for _ in range(pop_size):
        config = NeuralNetworkConfig()
        autoencoder = Autoencoder(config, [layer_size, next_layer_size])
        population.append((autoencoder, float("inf")))

    return population

def create_child(auto_enc):
    """ Create a child through mutation

    Args:
        auto_enc: The parent autoencoder from which the child is generated.

    Returns:
        An (autoencoder, fitness) tuple that has a copy of the history of its parent. The
        weights are also restored from the paren autoencoder in the first
        training session. The configuration is also copied and then mutated.
    """
    auto_enc = auto_enc[0]
    new_conf = auto_enc.config.copy()
    new_conf.mutate()
    new_hist = auto_enc.history_saver.copy()

    new_ae = Autoencoder(new_conf, list(auto_enc.layer_sizes), auto_enc.save_path), float("inf")
    new_ae[0].history_saver = new_hist
    return new_ae

if __name__ == "__main__":
    main()
