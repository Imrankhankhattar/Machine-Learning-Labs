import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data=datasets.load_iris()
# print(data.keys())
# print(data["data"])
#print(data["feature_names"])
#print(data["target_names"])

#Creating a pandas dataframe from the data
df=pd.DataFrame(data["data"],columns=data["feature_names"])
df["target"]=data["target"]
print(data["target_names"])
#print(df.describe())
#Distributions
col="sepal length (cm)"
df[col].hist()
plt.suptitle(col)
#plt.show()
col="sepal width (cm)"
df[col].hist()
plt.suptitle(col)
#plt.show()
#Relationship with our data
df["target_name"]=df["target"].map({0:"setosa",1:"versicolor",2:"virginica"})
#print(df.head(5))
sns.relplot(x=col,y="target",hue="target_name",data=df)
plt.suptitle(col,y=1)
#plt.show()
#Exploratory Analysis Pairplot
sns.pairplot(df,hue="target_name")
#plt.show()
#Making the model
#......................................
df_train,df_test=train_test_split(df,test_size=0.25)#inplace=True for droping columns from orignal dataframe
#print(df_train.shape)
X_train=df_train.drop(columns=["target","target_name"]).values
Y_train=df_train["target"].values
# print(X_train)
#print(Y_train)
# print(df_train)
#................Base Line Question..........Our Model  must have greater than 33% Accuracy..............
#............Modeling Manually...................
def manual_prediction(petal_length):
    if (petal_length < 2.5):
        return 0 
    elif (petal_length < 4.8):
        return 1
    else:
        return 2
#print(X_train[:,2])     
#Y_result=[manual_prediction(val) for val in X_train[:,2]]
Y_result=np.array([manual_prediction(val) for val in X_train[:,2]])
# print(np.mean(Y_result==Y_train))
#..............................................................................
model=LogisticRegression(max_iter=100)
Xt,Xv,Yt,Yv=train_test_split(X_train,Y_train,test_size=0.20)
model.fit(Xt,Yt)
y_predict=model.predict(Xv)
print(np.mean(y_predict==Yv))
print(model.score(Xv,Yv))#Alternative of upper two command