import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

ds = pd.read_csv('TATAMOTORS.csv')

#print(ds.head())
#print(ds.shape)
#print(ds.describe())

X = ds.iloc[:, 1:2].values

print(X)

y = ds.iloc[: , 4].values
print(y)

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)