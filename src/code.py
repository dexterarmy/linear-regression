import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
import json
import matplotlib.pyplot as plt

Boston=pd.read_csv('./data/Boston.csv')
y = Boston[['medv']]
x = Boston[['crim']]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
lr = LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_test.head()
y_pred[0:5]
model_1=mean_squared_error(y_test,y_pred)
print(model_1)

plt.scatter(x_test, y_test, color='black')
plt.plot(x_test, y_pred, color='blue', linewidth=3)
plt.xlabel("crim")
plt.ylabel("medv")
plt.title("Linear Regression (crim vs medv)")
plt.savefig("model_1.png")

x = Boston[['lstat']]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
lr2 = LinearRegression()
lr2.fit(x_train,y_train)
y_pred2=lr2.predict(x_test)
model_2=mean_squared_error(y_test,y_pred2)
print(model_2)


plt.scatter(x_test, y_test, color='black')
plt.plot(x_test, y_pred2, color='red', linewidth=3)
plt.xlabel("lstat")
plt.ylabel("medv")
plt.title("Linear Regression (lstat vs medv)")
plt.savefig("model_2.png")

with open("metrics.txt", 'w') as outfile:
    json.dump({"model_1" : model_1, "model_2": model_2}, outfile)

