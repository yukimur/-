
import sklearn
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
from sklearn.decomposition import PCA

# 文本特征抽取
# 独热编码+次数
data = ['ds fdf gfgfd gfgfd','ghg hbf rt','ds fdf gfgfd']
trans = CountVectorizer(stop_words=[])
a = trans.fit_transform(data)
print(a.toarray())

# tfidf
trans = TfidfVectorizer(stop_words=[])
a = trans.fit_transform(data)
print(a.toarray())

# 特征预处理
# 归一化 鲁棒性较差
data = [[1.2,2.3,4.6],[0.2,0.9,0.5]]
trans = MinMaxScaler(feature_range=[-1,1])
a = trans.fit_transform(data)
print(a)
# 标准化 鲁棒性较强
trans = StandardScaler()
a = trans.fit_transform(data)
print(a)

# 降维
# 特征选择
    # filter过滤式
        # 方差选择法：低方差特征过滤
        # 相关系数：特征与特征相关程度 皮尔逊相关系数
    # embedded
        # 决策树
        # 正则化
        # 深度学习
# 方差过滤
data = [[1.2,2.3,4.6],[1.2,0.9,0.5]]
trans = VarianceThreshold(threshold=1)
a = trans.fit_transform(data)
print(a)

# 相关系数
import numpy as np
a = pearsonr(data[0],data[1])
print(a)

# 主成分分析
# n_components
    # 小数：保留百分之几的信息
    # 整数：减少到多少特征
data = [[2,8,4,5],[6,3,0,8],[5,4,9,1]]
trans = PCA(n_components=0.95)
a = trans.fit_transform(data)
print(a)