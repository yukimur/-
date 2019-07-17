
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer


# 获取数据
train = pd.read_csv('titanic/train.csv')
test = pd.read_csv('titanic/test.csv')
print(train.head())
x_train = train[['Pclass','Age','Sex']]
y_train = train['Survived']
x_test = test[['Pclass','Age','Sex']]
y_test = test['Survived']
print(x_train.head())

# 缺失值处理
x_train["Age"].fillna(x_train["Age"].mean(),inplace=True)
x_test["Age"].fillna(x_train["Age"].mean(),inplace=True)

x_train = x_train.to_dict(orient="records")
x_test = x_test.to_dict(orient="records")
# 字典特征抽取
trans = DictVectorizer()
x_train = trans.fit_transform(x_train)
x_test = trans.transform(x_test)
# 决策树
estimator = DecisionTreeClassifier(criterion='entropy',max_depth=5)
    # criterion:衡量方式
    # max_depth:深度最大
estimator.fit(x_train,y_train)

# 模型评估
y_pridict = estimator.predict(x_test)
print(y_pridict)
score = estimator.score(x_test,y_test)
print(score)