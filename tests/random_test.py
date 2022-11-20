import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_data = range(10)
y_data = range(10)
print("---------------")
print("random seed or states are same, the dataset splits created will be exactly same.")
for i in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.3,random_state = 0) 
    print(y_test)

print("*"*30)
print("------------------")
print("test case that verified that if random seed or states are different, the dataset splits created will not be exactly same")
for i in range(5): 
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.3,random_state = None)
    print(y_test)
