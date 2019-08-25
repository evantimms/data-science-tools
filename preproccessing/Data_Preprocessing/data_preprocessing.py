#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 12:27:25 2019

@author: evantimms
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# use for applying feature scaling
#from sklearn.preprocessing import StandardScaler
#sc_x = StandardScaler()
#X_train = sc_x.fit_transform(X_train)
#X_test = sc_x.transform(X_test)