import matplotlib.pyplot as plt
from sklearn import linear_model
x=[[75],[35],[70],[45],[85],[65],[90],[55],[25]]
y=[3.3,0,3,1.2,3.7,2.7,4,2,0]
reg=linear_model.LinearRegression()
reg.fit(x,y)
plt.plot(x,y,color="r",marker="o")
plt.plot(x,reg.predict(x),color="b",marker="x")
plt.show()
print(reg.predict([[75]]))
