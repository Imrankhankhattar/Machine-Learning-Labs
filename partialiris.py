import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data=datasets.load_iris()
df=pd.DataFrame(data["data"],columns=data["feature_names"])
# print(df.describe())
X_train=df.drop(columns=["petal width (cm)"]).values
Y_train=df["petal width (cm)"].values
model=LinearRegression()
Xt,Xv,Yt,Yv=train_test_split(X_train,Y_train,test_size=0.20)
model.fit(Xt,Yt)
y_predict=model.predict(Xv)
# print(y_predict)
# print(Yv)
#print(np.mean(y_predict==Yv))
print(model.score(Xv,Yv))#Alternative of upper two command