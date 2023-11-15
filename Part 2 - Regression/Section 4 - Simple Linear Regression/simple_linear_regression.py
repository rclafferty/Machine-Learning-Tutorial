############################
# SIMPLE LINEAR REGRESSION #
############################

# Importing the libraries #
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Importing the dataset #
dataset = pd.read_csv('Salary_Data.csv')

data_features = dataset.iloc[:, :-1].values # Take all columns EXCEPT last one
dependent_var = dataset.iloc[:, -1].values # Take ONLY last column

# Splitting the dataset into the training set and testing set #
# 80% in training set, 20% in test set
features_train, features_test, dependent_train, dependent_test = train_test_split(data_features, dependent_var, test_size = 0.2, random_state = 1)

# Training the simple linear regression model on the training set #
regressor = LinearRegression()
regressor.fit(features_train, dependent_train)

# Predicting the test set results #
predict_test = regressor.predict(features_test)
predict_train = regressor.predict(features_train)

# Visualizing the training set results #
plt.scatter(features_train, dependent_train, color='red')
plt.plot(features_train, predict_train, color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualizing the test set results #
plt.scatter(features_test, dependent_test, color='red')
plt.plot(features_train, predict_train, color='blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')

prediction_single_x = 9
prediction_single_y = regressor.predict([[prediction_single_x]])
print(prediction_single_y)

plt.scatter(prediction_single_x, prediction_single_y, color='green')

plt.show()

# Getting the final linear regression equation with the values of the coefficients #
b0 = regressor.intercept_
b1 = regressor.coef_

print(regressor.coef_)
print(regressor.intercept_)
print(f'y = {b1[0]:.2f}x + {b0:.2f}')

