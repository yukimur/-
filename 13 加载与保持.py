
from sklearn.externals import joblib

e = None
# 保存
joblib.dump(e,'t.pkl')
# 加载
e = joblib.load('t.pkl')