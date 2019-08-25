#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 18:56:20 2019

@author: evantimms
"""

# Regression Template
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, ].values
Y = dataset.iloc[:, ].values

# Fitting the Regression model to the dataset

# Predicting a value with the regression
y_pred = regressor.predict(6.5)

# Visualize
plt.scatter(X, Y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Polynomial Linear Regression on Polynomial DS')
plt.xlabel('Position Level')
plt.ylabel('Position Salary')
plt.show()
