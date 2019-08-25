# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# Take care of missing data
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values = np.nan, strategy = 'mean') # exchange mean values for mean of columns
imp = imp.fit(X[:, 1:3])
X[:, 1:3] = imp.transform(X[:, 1:3])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# encode the x using a combination of the label transformer to create a digit 
# and the hot encoder to split the columns so that the ml model does 
# not assign differing weights
labelencoder_x = LabelEncoder()
X[:, 0] = labelencoder_x.fit_transform(X[:, 0])
hotenc = OneHotEncoder(categorical_features = [0 ])
X = hotenc.fit_transform(X).toarray()
# encoding the y category
labelencoder_y = LabelEncoder()
Y = labelencoder_y.fit_transform(Y)

# Splitting the dataset int o the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Applying feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)
# do not need to apply feature scaling to y as this is only categorical
# will need to do this for regression however
