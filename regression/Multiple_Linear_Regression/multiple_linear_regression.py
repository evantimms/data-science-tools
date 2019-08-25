# Multiple Linear Regression
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values


# Encode the categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
ohe = OneHotEncoder(categorical_features=[3])
X[:, 3] = le.fit_transform(X[:, 3])
X = ohe.fit_transform(X).toarray()

# Remove n-1 dummy variables
X = X[:, 1:]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Fit the mlr model to the training set
reg = LinearRegression()
reg.fit(X_train, Y_train)

# Predict the test set results
y_pred = reg.predict(X_test)

# Using BE to build the optimal model
# Add the value for the intercept by adding x0 as coloumn of 1s
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)

# Initalize X_opt with all independent variables and significance level
X_opt = X
sl = 0.05
cols = [0, 1, 2, 3, 4, 5]
highest = 1

# Get the optimal least squares regressor
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
print(regressor_OLS.summary())

while highest > sl:
    p_values = regressor_OLS.pvalues
    prev = regressor_OLS
    # get the highest p-value
    highest = max(p_values).astype(float)
    i = np.where(p_values == highest)[0][0]
    if highest > sl:
            cols.pop(i)
            X_opt = X[:, cols]
    # refit the regressor
    regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()

# Keep the previous model
regressor_OLS = prev
print(regressor_OLS.summary())

