"""Implementation of the co-ranking matrix and derived metrics
Author: Samuel Jackson
Modified by: Tim Silhan

The MIT License (MIT)

Copyright (c) 2015 Samuel Jackson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
from scipy.spatial import distance
from sklearn import manifold, datasets


def coranking_matrix(high_data, low_data):
    """Generate a co-ranking matrix from two data frames of high and low dimensional data.
    Args:
        high_data: DataFrame containing the higher dimensional data.
        low_data: DataFrame containing the lower dimensional data.
    Returns:
        The co-ranking matrix of the two data sets.
    """
    n, m = high_data.shape
    high_distance = pairwise_distances(high_data)
    low_distance = pairwise_distances(low_data)

    high_ranking = high_distance.argsort(axis=1).argsort(axis=1)
    low_ranking = low_distance.argsort(axis=1).argsort(axis=1)

    Q, xedges, yedges = np.histogram2d(high_ranking.flatten(),
                                       low_ranking.flatten(),
                                       bins=n)

    # Q = Q[1:, 1:]  # remove rankings which correspond to themselves
    return Q


def pairwise_distances(X):
    """ Computes the matrix of distances between the values in X """
    return distance.squareform(distance.pdist(X))

def lcmc(Q, K):
    """ The local continuity meta-criteria measures the number of mild
    intrusions and extrusions. This can be thought of as a measure of the
    number of true postives.
    Args:
        Q: the co-ranking matrix to calculate continuity from
        k (int): the number of neighbours to use.
    Returns:
        The LCMC metric for the given K
    """
    n = Q.shape[0]
    summation = 0.0

    for k in range(K):
        for l in range(K):
            summation += Q[k, l]

    return (K / (1. - n)) + (1. / (n*K)) * summation

def qnx(Q, K):
    """ The local continuity meta-criteria measures the number of mild
    intrusions and extrusions. This can be thought of as a measure of the
    number of true postives.
    Args:
        Q: the co-ranking matrix to calculate continuity from
        k (int): the number of neighbours to use.
    Returns:
        The Qnx metric for the given K

    Note: https://arxiv.org/pdf/1110.3917.pdf
    """
    n = Q.shape[0]
    summation = 0.0

    for k in range(K):
        for l in range(K):
            summation += Q[k, l]

    return (1. / (n*K)) * summation


def _make_datasets():
    high_data, color \
        = datasets.samples_generator.make_swiss_roll(n_samples=300,
                                                     random_state=1)

    isomap = manifold.Isomap(n_neighbors=12, n_components=2)
    low_data = isomap.fit_transform(high_data)

    return high_data, low_data


def _test_lcmc():
    high_data, low_data = _make_datasets()
    Q = coranking_matrix(high_data, low_data)
    l = lcmc(Q, 5)
    print(l)

    # Only true with co-ranking matrix with ranks that correspond to themselves
    # removed
    # assert_almost_equal(l, 0.377, places=3)

if __name__ == "__main__":
    _test_lcmc()
