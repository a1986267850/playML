import numpy as np

class module_Selection:
    def __init__(self, rate=.2, seed=None):
        self.rate = rate
        self.seed = seed

    def train_test_split(self, x_data, y_data):
        if self.seed:
            np.random.seed(self.seed)
        rands = np.random.permutation(len(x_data))
        train_size = int(len(x_data)*self.rate)
        x_train = x_data[rands[train_size:]]
        y_train = y_data[rands[train_size:]]
        x_test = x_data[rands[:train_size]]
        y_test = y_data[rands[:train_size:]]
        return x_train, y_train, x_test, y_test