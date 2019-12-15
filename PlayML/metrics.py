import numpy as np


def acuatlly_score(self,y_test, y_predict):
    return np.sum(y_test == y_predict)/len(y_test)