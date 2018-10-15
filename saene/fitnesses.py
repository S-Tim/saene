""" Fitnes functions for evaluating an autoencoder

Author: Tim Silhan
"""

import numpy as np
from coranking import coranking_matrix, lcmc

def lcmc_fitness(individual, data, test=False):
    """ Calculates the local continuity meta criterion for the given *individual* and *data*

    The LCMC metric is calculated for use in the fitness evaluation of the
    individual in the population.

    Args:
        individual: Autoencoder for which the metric is calculated.
        data: A function that provides the input data for the network.

    Returns:
        LCMC fitness value
    """
    # Get test data
    if test:
        batch_x, _ = data.test.next_batch(50)
    else:
        batch_x, _ = data.validation.next_batch(50)

    # Encode test data
    prediction, _ = individual.run_encode_session(batch_x)
    # calculate lcmc value
    matrix = coranking_matrix(np.array(batch_x), np.array(prediction))
    lcmc_value = lcmc(matrix, 25)
    # insert value into population fitness
    return lcmc_value

def qnx_fitness(individual, data, test=False):
    """ Calculates the Qnx metric for the given *individual* and *data*

    The Qnx metric is calculated for use in the fitness evaluation of the
    individual in the population.

    Args:
        individual: Autoencoder for which the metric is calculated.
        data: A function that provides the input data for the network.

    Returns:
        Qnx fitness value
    """
    # Get test data
    if test:
        batch_x, _ = data.test.next_batch(50)
    else:
        batch_x, _ = data.validation.next_batch(50)

    # Encode test data
    prediction, _ = individual.run_encode_session(batch_x)
    # calculate lcmc value
    matrix = coranking_matrix(np.array(batch_x), np.array(prediction))
    lcmc_value = lcmc(matrix, 25)
    # insert value into population fitness
    return lcmc_value

def mse_fitness(individual, data, test=False):
    """ Calculates the mean square error of the reconstructed and original images

    The mean square error is calculated on a validation set that is separate
    from the test and training sets.

    Args:
        individual: Autoencoder for which the metric is calculated.
        data: A function that provides the input data for the network.

    Returns:
        MSE fitness value
    """
    # Get test data
    if test:
        batch_x, _ = data.test.next_batch(50)
    else:
        batch_x, _ = data.validation.next_batch(50)

    # Encode test data to get the loss_value
    _, loss = individual.run_encode_session(batch_x)

    return loss
