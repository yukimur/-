
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# kmeans评估指标
    # 轮廓系数 sci = (bi-ai)/max(bi,ai)
    # 取值范围在[-1,1],越接近1越好

silhouette_score(data,y_pred)