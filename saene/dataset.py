"""
Dataset class to mimic the interface of the tensorflow.examples.tutorials.mnist datasets.

Author: Tim Silhan
"""

import numpy as np

class Dataset:
    """ Manages a dataset

    Datasets are created from a list (preferably a numpy array) of features
    and labels. The data is then shuffled and repeated infinitely, shuffling
    again after every epoch.

    Attributes:
        features: List of features of the dataset
        labels: List of labels of the dataset
        current_index: Current position in the dataset
    """

    def __init__(self, features, labels):
        """ Initializes the dataset

        Args:
            features: List of features
            labesls: List of labels
        """
        self.features = features
        self.labels = labels
        self.current_index = 0

        self.shuffle()

    def next_batch(self, batch_size):
        """ Fetches the next batch of data from the dataset

            Args:
                batch_size: *batch_size* features and labels are returned as a batch

            Returns:
                Tuple of features and labels each batched to *batch_size*
        """
        new_index = self.current_index + batch_size

        if new_index <= len(self.features):
            batch_features = np.array(self.features[self.current_index:new_index])
            batch_labels = np.array(self.labels[self.current_index:new_index])
            self.current_index = new_index
        else:
            # Get remaining samples
            batch_features = np.array(self.features[self.current_index:len(self.features)])
            batch_labels = np.array(self.labels[self.current_index:len(self.labels)])

            self.shuffle()
            new_index %= len(self.features)

            # Append from the newly shuffled dataset
            batch_features = np.append(batch_features, np.array(self.features[0:new_index]), axis=0)
            batch_labels = np.append(batch_labels, np.array(self.labels[0:new_index]), axis=0)
            self.current_index = new_index

        return batch_features, batch_labels

    def shuffle(self):
        """ Shuffles the dataset """
        perm = np.arange(len(self.features))
        np.random.shuffle(perm)
        self.features = self.features[perm]
        self.labels = self.labels[perm]


if __name__ == "__main__":
    FEATURES = np.array(list(range(50)))
    LABELS = np.array(list(range(50)))
    DS = Dataset(FEATURES, LABELS)

    print(DS.next_batch(10))
    print(DS.next_batch(40))
    print(DS.next_batch(10))
    print(DS.next_batch(0))
    print(DS.next_batch(9))
    print(DS.next_batch(50))
