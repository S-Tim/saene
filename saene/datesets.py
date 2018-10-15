"""
Splits one dataset into train, test and validation datasets

Author: Tim Silhan
"""

import numpy as np
from dataset import Dataset

class Datasets:
    """ Reads a dataset from a txt file and parses it to *Dataset* instances

        Attributes:
            train: Training dataset
            test: Test dataset
            validation: Validation dataset
    """

    def __init__(self, path, train_size, validation_size):
        """ Reads a dataset from path and split it into train, test and validation sets

            Args:
                path: Path to the file containing the dataset. It has to be
                    colon-separated csv-style with no header.
                train_size: The size of the training dataset
                validation_size: The size of the validation dataset
        """
        dataset = []
        # Read dataset from file and parse each line to a numpy array
        with open(path) as file:
            dataset = np.array([self._parse_line(line) for line in file.readlines()])

        # Split the dataset into train, test and validate sets
        train_dataset = dataset[:train_size]
        validation_dataset = dataset[train_size:train_size+validation_size]
        test_dataset = dataset[train_size+validation_size:]

        print("Data set size: ", len(dataset))
        print("Training set size: ", len(train_dataset))
        print("Test set size: ", len(test_dataset))
        print("Validation set size: ", len(validation_dataset))

        self.train = Dataset(*self._split_labels(train_dataset))
        self.test = Dataset(*self._split_labels(test_dataset))
        self.validation = Dataset(*self._split_labels(validation_dataset))

    def _parse_line(self, line):
        return np.array([float(x) for x in line.split(",")])

    def _split_labels(self, dataset):
        labels = np.array(dataset[:, :1])
        features = np.array(dataset[:, 1:])

        return features, labels

if __name__ == "__main__":
    TRAIN_SIZE = 453715
    TEST_SIZE = 51630
    VAL_SIZE = 10000

    DS = Datasets("./data/year_prediction_msd/YearPredictionMSD.txt", TRAIN_SIZE, VAL_SIZE)
