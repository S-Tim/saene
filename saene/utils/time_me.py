""" Simple timing annotation """

import time
from functools import wraps

def time_me(func):
    """ Times the function given as parameter """

    @wraps(func)
    def wrapper(*args, **kwargs):
        """ Wraps the original function with the timing """
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        print("The function {} took {}s to run".format(func.__name__, end - start))

        return result

    return wrapper

if __name__ == "__main__":
    @time_me
    def some_function():
        """ Example function that takes some time """
        result = 0
        for i in range(100000):
            result += i
        return result

    print(some_function())
