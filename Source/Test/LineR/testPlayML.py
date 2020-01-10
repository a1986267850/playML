'''
Created on 2020年1月10日

@author: Administrator
'''
import numpy as np
import matplotlib.pyplot as plt
from ML.Line.LinearRegression import LinearRegression as linear
from sklearn import datasets
from ML.Line.MLinearRegression import MLinearRegression

from ML.KNN.module_Selection import module_Selection as PredictTest

if __name__ == '__main__':
    boston = datasets.load_boston()
    x = boston.data
    y = boston.target
    pt = PredictTest(.6)
    x,y,x_test,y_test = pt.train_test_split(x, y)
    reg = MLinearRegression()
    reg.fit(x, y)
    print(reg.inte,reg.coef)
    print(reg.theta)
    
    y_predict = reg.predict(x_test)
    
    print("score:",reg.score(y_test, y_predict))
    print()
    pass
def one():
    x = np.array([1.,2.,3.,4.,5.])
    y = np.array([1.,3.,2.,3.,5.])
    
    lin = linear()
    lin.fit(x, y)
    y_predict = lin.predict(x)
    plt.scatter(x,y)
    plt.plot(x,y_predict,'r')
    plt.axis([0,6,0,6])
    plt.show()
def lineR():
    bostn = datasets.load_boston()
    x = bostn.data[:,5]
    y = bostn.target
    max = np.max(y)
    x = x[y<max]
    y = y[y<max]
    plt.scatter(x,y)
    lin = linear()
    lin.fit(x,y)
    print(lin.a,lin.b)
    y_predict = lin.predict(x)
    plt.plot(x,y_predict,color='r')
    plt.show()
    #print("MSE:",lin.MSE(y, y_predict))
    #print("RMSE:",lin.RMSE(y, y_predict))
    #print("MAE:",lin.MAE(y, y_predict))
    print("s_quar:",lin.score(y, y_predict))
