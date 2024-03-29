# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

# Splitting into training and testing datasets is not needed here as there is not
# enough information with only ten columns

# Linear Regression as example to compare to
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

# Polynomial Regression model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 6)
X_poly = poly_reg.fit_transform(X)

# Create the new linear regression using the new fit
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, Y)

# Visualize the first linear regression
plt.scatter(X, Y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Linear Regression on Polynomial DS')
plt.xlabel('Position Level')
plt.ylabel('Position Salary')
plt.show()

# Visualize the PLR
X_grid = np.arange(min(X), max(X), 0.1) 
X_grid = X_grid.reshape(len(X_grid),1)

plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Polynomial Linear Regression on Polynomial DS')
plt.xlabel('Position Level')
plt.ylabel('Position Salary')
plt.show()

# Predicting a salary with the Linear Regression
pred_salary = lin_reg.predict([[6.5]])

# Predicitng a salary with the PLR
pred_salary_PLR = lin_reg2.predict(poly_reg.fit_transform([[6.5]])) 
