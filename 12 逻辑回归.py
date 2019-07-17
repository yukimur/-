
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# 获取数据
data = pd.read_csv('data.csv')
print(data.keys())
# 缺失值處理
data = data.iloc[:,1:-1].replace(to_replace="?",value=np.nan)
data.dropna(inplace=True)   # 删除缺失样本
print(data.isnull().any())

# 劃分數據集
x_train,x_test,y_train,y_test = train_test_split(data.iloc[:,2:-1],data["diagnosis"])
# 特徵工程
trans = StandardScaler()
x_train = trans.fit_transform(x_train)
x_test = trans.transform(x_test)
# 逻辑回归
estimator = LogisticRegression()
estimator.fit(x_train,y_train)
# 模型评估
print(estimator.coef_)
print(estimator.intercept_)
score = estimator.score(x_test,y_test)
print(score)