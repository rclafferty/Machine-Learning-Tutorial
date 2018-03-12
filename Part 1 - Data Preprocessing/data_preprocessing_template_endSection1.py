# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
    # [all of the rows, all of the columns EXCEPT the last one]
Y = dataset.iloc[:, 3].values

# Taking care of missing data
np.set_printoptions(threshold = np.nan)
from sklearn.preprocessing import Imputer
# imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0, verbose = 0, copy = True)
# CTRL+I to inspect the class
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3]) # Lowerbound is inclusive, Upperbound is exclusive
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])
# Dummy encoding - X
oneHotEncoder = OneHotEncoder(categorical_features = [0])
X = oneHotEncoder.fit_transform(X).toarray() # Already specified the first column in the categorical_features
# Dummy encoding - Y
labelEncoder_Y = LabelEncoder()
Y = labelEncoder_Y.fit_transform(Y)

# Splitting the dataset into the training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)