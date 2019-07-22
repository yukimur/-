
from sklearn.linear_model import Ridge,SGDRegressor # 都有L2，但前者有sag
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# 减少过拟合
    # 正则化
        # L1
        # L2

# 获取数据集
data = load_boston()
# 划分数据集
x_train,x_test,y_train,y_test= train_test_split(data.data,data.target,random_state=22)
# 特征工程
trans = StandardScaler()
x_train = trans.fit_transform(x_train)
x_test = trans.transform(x_test)
# 线性回归
estimator = Ridge()
estimator.fit(x_train,y_train)

# 模型评估:用均方误差衡量
print(estimator.coef_)      # 回归系数
print(estimator.intercept_) # 偏置
y_pred = estimator.predict(x_test)
score = mean_squared_error(y_test,y_pred)
print(score)