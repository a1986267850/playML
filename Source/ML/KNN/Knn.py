import numpy as np
from math import sqrt
from collections import Counter


class Knn:

    def __init__(self, k):
        self.k = k

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x):
        pre = [self.predict_(_x) for _x in x]
        return np.array(pre)

    def predict_(self, _x):
        distance = [sqrt(np.sum((_x-i)**2)) for i in self.x_train]
        rose = np.argsort(distance)[:self.k]
        return Counter(self.y_train[rose]).most_common()[0][0]

    def score(self, x_test, y_test):
        y_predict = self.predict(x_test)
        return np.sum(y_predict == y_test)/len(y_test)