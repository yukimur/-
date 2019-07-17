
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# knn特点
    # 简单易于实现，无需训练
    # 懒惰算法，计算大，内存占用大
    # 必须指定K值
    # 适合小数据场景
# 距离计算
    # 欧式距离：平方差
    # 曼哈顿距离： 绝对值距离
    # 明可夫斯基距离

def knn():
    # 获取数据
    iris = load_iris()
    # 划分数据集
    x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,random_state=10)
    # 标准化
    trans = StandardScaler()
    x_train = trans.fit_transform(x_train)
    x_test = trans.transform(x_test)    # 注意

    # knn
    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train,y_train)

    # 模型评估
    y_predict = estimator.predict(x_test)
    print(y_predict==y_test)
    # 准确率
    score = estimator.score(x_test,y_test)
    print(score)

if __name__ == "__main__":
    knn()