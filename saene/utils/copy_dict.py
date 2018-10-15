""" Utility to copy dictionaries that also have lists as values

Author: Tim Silhan
"""

import numpy as np

def copy_dict(original):
    """ Copies the *original* dictionary

    This function makes sure that lists and numpy arrays are copied as well
    instead of only referencing the original list in the copy.

    Args:
        original: Dictionary that should be copied

    Returns:
        A copy of the *original* dictionary
    """
    copied = original.copy()

    for key in copied.keys():
        if isinstance(copied[key], list):
            copied[key] = copied[key][:]
        elif isinstance(copied[key], np.ndarray):
            copied[key] = np.copy(copied[key])

    return copied

if __name__ == "__main__":
    TEST = {"a" : 1, "b" : "hello", "c" : [1, 2, 3], "d" : ["he", "lo"]}
    COPIED = copy_dict(TEST)

    TEST["d"][0] = "world"
    TEST["c"][2] = 4

    assert TEST["d"][0] != COPIED["b"][0]
    assert TEST["c"][2] != COPIED["c"][2]
    assert TEST["c"][1] == COPIED["c"][1]
    assert TEST["b"] == COPIED["b"]
