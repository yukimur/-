
from sklearn.datasets import fetch_20newsgroups,load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


# 拉普拉斯平滑
    # 分子+1/分母+特招数

# 获取数据
train = load_files(container_path='20news-bydate-train',encoding='utf-8')
test = load_files(container_path='20news-bydate-test',encoding='utf-8')
# 划分数据
x_train,x_test,y_train,y_test = train.data,train.target,test.data,test.target
# 特征工程
trans = TfidfVectorizer()
x_train = trans.fit_transform(x_train)
x_test = trans.transform(x_test)
# 朴素贝叶斯
estimator = MultinomialNB(alpha=1.0)
estimator.fit(x_train,y_train)
# 模型评估
y_predict = estimator.predict(x_test,y_test)
print(y_predict)
score = estimator.score(x_test,y_test)
print(score)