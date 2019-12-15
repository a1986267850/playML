import numpy as np
class StandScalar:
    def __init__(self):
        self.mean_ = None
        self.std = None
    def fit(self,x_train):
        assert x_train.dina == 2,"the shape must be two"
        restX = np.array(x_train, dtype=float)
        self.mean_ =  [np.mean(restX[:,i]) for i in range(restX.shape[1])]
        self.std_ = [np.std(restX[:,i]) for i in range(restX.shape[1])]
        return self
    def transform(self,x):
        resX = np.array(x ,dtype=float)
        for col in range(resX.shape[1]):
            resX[:,col] = (resX[:,col] - self.mean_[col]) / self.std_[col]
        return resX