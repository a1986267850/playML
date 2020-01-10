from ML.KNN.module_Selection import module_Selection
from ML.KNN.Knn import Knn
import ML.KNN.metrics as metrics

import matplotlib
from sklearn import datasets
from matplotlib import pyplot as mpl

if __name__ == '__main__':
    digist = datasets.load_digits()
    datas = digist.data
    target = digist.target
    # data = datas[666]
    # result = target[666]
    # mpl.imshow(data.reshape(8, 8), cmap=matplotlib.cm.binary)
    # mpl.show()
    selection = module_Selection(rate=.2, seed=666)
    x_train, y_train, x_test, y_test = selection.train_test_split(datas, target)
    knn = Knn(k=6)
    knn.fit(x_train, y_train)
    y_predict = knn.predict(x_test)
    # score = metrics().acuatlly_score(y_test, y_predict)
    # print(score)
    score = knn.score(x_test,y_test)
    print(score)