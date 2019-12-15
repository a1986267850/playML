from PlayML.module_Selection import module_Selection
from PlayML.Knn import Knn
from sklearn import datasets
if __name__ == '__main__':
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    module = module_Selection(rate=0.4, seed=666)
    x_train, y_train, x_test, y_test = module.train_test_split(x,y)
    # print(len(x_train))
    # print(len(y_train))
    knn = Knn(k=6)
    knn.fit(x_train,y_train)
    y_pre = knn.predict(x_test)
    print('预测结果:{value}'.format(value=y_pre))
    print('预测准确率:{0}%'.format(int(sum(y_pre==y_test)/len(x_test)*100)))