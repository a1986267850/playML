'''
Created on 2020年1月10日

@author: Administrator
'''
import numpy as np
from math import sqrt
class MLinearRegression:
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.coef = None
        self.inte = None
        self.theta = None
    def fit(self,x,y):
        Xb = np.hstack([np.ones((len(x),1)),x])
        self.theta = np.linalg.inv(Xb.T.dot(Xb)).dot(Xb.T.dot(y))
        self.inte = self.theta[0]
        self.coef = self.theta[1:]
    def predict(self,x_test):
        return np.hstack([np.ones((len(x_test),1)),x_test]).dot(self.theta)
    def score(self,y,y_predict):
        return 1-np.sum((y-y_predict)**2)/len(y_predict) / np.var(y)