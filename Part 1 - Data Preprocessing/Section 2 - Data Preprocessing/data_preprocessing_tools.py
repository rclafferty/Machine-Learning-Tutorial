############################
# DATA PREPROCESSING TOOLS #
############################

# Importing the libraries #
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Importing the dataset #
dataset = pd.read_csv('Data.csv')

data_features = dataset.iloc[:, :-1].values # Take all columns EXCEPT last one
dependent_var = dataset.iloc[:, -1].values # Take ONLY last column

# Taking care of missing data #
# One option: Delete missing rows (only works for large data sets)
# "Right" option: Replace values with average of all other data in that column
# Will use SimpleImputer to do the "right" option
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Fit using the NUMERICAL columns only
imputer.fit(data_features[:, 1:3]) # Upper bound is EXCLUDED, so this is [1, 3)
data_features[:, 1:3] = imputer.transform(data_features[:, 1:3]) # Transform and then replace the existing dataset with updated values

# Encoding categorical data #
# Title -- no code

# Encoding the independent variables #
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
data_features = ct.fit_transform(data_features)

# Encoding the dependent variable #
le = LabelEncoder()
dependent_var = le.fit_transform(dependent_var)

# print(data_features)
# print(dependent_var)

# Splitting the dataset into the training set and testing set #
# 80% in training set, 20% in test set
features_train, features_test, dependent_train, dependent_test = train_test_split(data_features, dependent_var, test_size = 0.2, random_state = 1)

print(features_train)
print(features_test)
print(dependent_train)
print(dependent_test)

# Feature scaling #
# Avoid some features being dominated by other features and imbalance the model
# Not always necessary
# Two ways: Standardization or Normalization
# - Normalization = Recommended when you have a normal distribution among all features
# - Standardization = Useful for all data sets
ss = StandardScaler()
features_train[:, -2:] = ss.fit_transform(features_train[:, -2:])
features_test[:, -2:] = ss.fit_transform(features_test[:, -2:])

# print(data_features)
# print(dependent_var)