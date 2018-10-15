""" Configuration for the evolutionary algorithm

Author: Tim Silhan
"""

class EAConfig():
    """ Configuration of the evolutionary algorithm

    Attributes:
        pop_size: Individuals per generation
        gens_per_layer: Generations per layer
        selection_ratio: Ratio of parent generation that carries on to next generation
        layer_ratio: Ratio of new and old layer
        layer_size: Size of the currently smallest layer
        target_size: Maximum size of the last encoder layer
    """

    def __init__(self):
        self.pop_size = 10
        self.gens_per_layer = 10
        self.selection_ratio = 0.51
        self.layer_ratio = 0.5
        self.layer_size = 784
        self.target_size = 200

    def __str__(self):
        attributes = [entry for entry in self.__dict__ if not entry.startswith("__")]
        pairs = dict([(key, self.__getattribute__(key)) for key in attributes])

        representation = ""
        for key, val in pairs.items():
            representation += "{}: {}\n".format(str(key).ljust(15), val)

        return representation
    