
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

# 交叉验证：将数据分为n份，进行n次测试，每次更换不同的验证集，为了选择模型
# 网格搜索：为了选择参数

def knn_gscv():
    # 获取数据
    iris = load_iris()
    # 划分数据集
    x_train,x_test,y_train,y_test = train_test_split(iris.data,iris.target,random_state=10)
    # 标准化
    trans = StandardScaler()
    x_train = trans.fit_transform(x_train)
    x_test = trans.transform(x_test)    # 注意

    # knn
    estimator = KNeighborsClassifier()  # 不要K值
    # 加入网格搜索与交叉验证
    param_dict = {"n_neighbors":[1,3,5,7,9,11]}
    estimator = GridSearchCV(estimator,param_grid=param_dict,cv=10)
    estimator.fit(x_train,y_train)

    # 模型评估
    y_predict = estimator.predict(x_test)
    print(y_predict==y_test)
    # 准确率
    score = estimator.score(x_test,y_test)
    print(score)

    print(estimator.best_params_)    # 查看最佳参数
    print(estimator.best_score_)
    print(estimator.best_estimator_)
    print(estimator.cv_results_)    # 最佳交叉验证结果

knn_gscv()