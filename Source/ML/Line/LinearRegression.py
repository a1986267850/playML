'''
Created on 2020年1月10日

@author: Administrator
'''
import numpy as np
from math import sqrt


class LinearRegression:
    def __init__(self):
        self.a = None
        self.b = None
        
    def fit(self,x,y):
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        num = 0.0
        d = 0.0
        num += (x - x_mean).dot(y - y_mean)
        d += (x - x_mean).dot(x - x_mean)
        self.a = num / d 
        #print("%f,%f"%(num,d))
        self.b = y_mean - self.a * x_mean
    def predict(self,x_predict):
        return np.array([self.a*i+self.b for i in x_predict])
    def _MSE(self,y_truth,y_predict):
        return np.sum((y_predict-y_truth)**2) / len(y_predict)
    
    def MSE(self,y_truth,y_predict):
        return self._MSE(y_truth, y_predict)
    
    def RMSE(self,y_truth,y_predict):
        return sqrt(self._MSE(y_truth, y_predict))
    
    def MAE(self,y_truth,y_predict):
        return np.sum(np.absolute(y_truth-y_predict)) / len(y_predict)
    
    def _R_SQUARED(self,y_truth,y_predict):
        '''
        np.var = np.sum((mean_y - y_predict)**2 / len(y_predict))
        
        '''
        return 1- self._MSE(y_truth, y_predict)/np.var(y_truth)
        
         
    def score(self,y,y_predict):
        return self._R_SQUARED(y, y_predict)