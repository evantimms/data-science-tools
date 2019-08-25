# Simple linear regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predict the test results
y_pred = regressor.predict(X_test)

# Visualizing the training data
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training)')
plt.xlabel('Experience (yrs)')
plt.ylabel('Salary (USD)')
plt.show()

# Visualizing the testing data
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='green')
plt.title('Salary vs Experience (Testing)')
plt.xlabel('Experience (yrs)')
plt.ylabel('Salary (USD)')
plt.show()