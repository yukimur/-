
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.model_selection import train_test_split

# 信息熵
# 条件熵
# 信息增益
# 不需要标准化

# 获取数据
data = load_iris()
# 划分数据集
x_train,x_test,y_train,y_test = train_test_split(data.data,data.target,random_state=6)
# 决策树
estimator = DecisionTreeClassifier(criterion='entropy')
    # criterion:衡量方式
    # max_depth:深度最大
estimator.fit(x_train,y_train)

# 模型评估
y_pridict = estimator.predict(x_test)
print(y_pridict)
score = estimator.score(x_test,y_test)
print(score)

# 可视化
export_graphviz(estimator,out_file='tree.dot',feature_names=data.feature_names)