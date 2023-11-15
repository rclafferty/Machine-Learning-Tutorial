##############################
# MULTIPLE LINEAR REGRESSION #
##############################

# Importing the libraries #
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Importing the dataset #
dataset = pd.read_csv('50_Startups.csv')

data_features = dataset.iloc[:, :-1].values # Take all columns EXCEPT last one
dependent_var = dataset.iloc[:, -1].values # Take ONLY last column

# Encoding categorical data #
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
data_features = np.array(ct.fit_transform(data_features))

# Splittng the dataset into the training set and test set #
# 80% in training set, 20% in test set
features_train, features_test, dependent_train, dependent_test = train_test_split(data_features, dependent_var, test_size = 0.2, random_state = 1)

# Training the multiple linear rgression model on the training set #
regressor = LinearRegression()
regressor.fit(features_train, dependent_train)

# Predicting the test set results #
predict_test = regressor.predict(features_test) # y_pred
predict_train = regressor.predict(features_train)

np.set_printoptions(precision=2)
print(np.concatenate((predict_test.reshape(len(predict_test), 1), dependent_test.reshape(len(dependent_test), 1)), axis=1))


print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))
print(regressor.coef_)
print(regressor.intercept_)