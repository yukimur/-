
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import pandas as pd

data = pd.read_csv('facebook数据集/submissions_metadata.csv')
print(data.head())

# 过滤
data = data.query("x<2.5 & x> 2 & y<1.5 & y>1.0")
print(data)

# 处理时间特征
time_value = pd.to_datetime(data["time"],unit="s")
print(time_value)
date = pd.DatetimeIndex(data["time"])
print(date.day)

data["weekday"] = date.weekday
data["hour"] = date.hour

# 过滤签到次数少的地点
place_count = data.groupby("place_id").count()["row_id"]
data = data[data["place_id"].isin(place_count[place_count>3].index.values)]
x = data[["x","y","accuracy","day"]]
y = data["place_id"]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y)
