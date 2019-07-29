
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier

# 集成算法
    # bagging模型：并行模型，随机森林
    # boosting模型:串行模型，adaBoost,XgBoost
    # stacking:堆叠

# 随机森林是一个包含多个决策树的分类器，属于集成学习方法
# 随机
    # 训练集随机：bootstrap抽样
    # 特征值随机：从M个特征随机取m个
# 优点
    # 能处理维度高的数据，并且可以不作特征选择  feature_importance
    # 能看出哪些特征比较重要
    # 快，便于可视化

# 数据准备
train = pd.read_csv('titanic/train.csv')
test = pd.read_csv('titanic/test.csv')

x_train = train[['Pclass','Age','Sex']]
y_train = train['Survived']
x_test = test[['Pclass','Age','Sex']]
y_test = test['Survived']
# 缺失值处理
x_train["Age"].fillna(x_train["Age"].mean(),inplace=True)
x_test["Age"].fillna(x_train["Age"].mean(),inplace=True)

x_train = x_train.to_dict(orient="records")
x_test = x_test.to_dict(orient="records")
# 字典特征抽取
trans = DictVectorizer()
x_train = trans.fit_transform(x_train)
x_test = trans.transform(x_test)

# 参数准备
param_dict = {"n_estimators":[120,200,300,500,800,1200],
              "max_depth":[5,8,15,25,30]}
estimator = RandomForestClassifier()
estimator = GridSearchCV(estimator,param_grid=param_dict,cv=3)
estimator.fit(x_train,y_train)

# 模型评估
y_predict = estimator.predict(x_test)
print(y_predict == y_test)
# 准确率
score = estimator.score(x_test, y_test)
print(score)

print(estimator.best_params_)  # 查看最佳参数
print(estimator.best_score_)
print(estimator.best_estimator_)
print(estimator.cv_results_)  # 最佳交叉验证结果