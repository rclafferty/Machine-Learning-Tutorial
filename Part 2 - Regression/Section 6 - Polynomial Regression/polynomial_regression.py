#########################
# POLYNOMIAL REGRESSION #
#########################

# Importing the libraries #
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Importing the dataset #
dataset = pd.read_csv('Position_Salaries.csv')

data_features = dataset.iloc[:, 1:-1].values # NEW: Ignore the first column (extra -- just labels)
dependent_var = dataset.iloc[:, -1].values

# Training the linear regression model on the whole dataset #
linear_regressor = LinearRegression()
linear_regressor.fit(data_features, dependent_var) # Did not split the dataset -- training on whole dataset

# Training the polynomial regression model on the whole dataset #
deg = 4
polynomial_regressor = PolynomialFeatures(degree = deg)
features_poly = polynomial_regressor.fit_transform(data_features)

linear_regressor_2 = LinearRegression()
linear_regressor_2.fit(features_poly, dependent_var)

# Visualizing the linear regression results #
plt.scatter(data_features, dependent_var, color = 'red') # real results
plt.plot(data_features, linear_regressor.predict(data_features), color = 'blue') # predicted results
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualizing the polynomial regression results #
plt.scatter(data_features, dependent_var, color = 'red') # real results
plt.plot(data_features, linear_regressor_2.predict(features_poly), color = 'blue') # predicted results
plt.title(f'Truth or Bluff (Polynomial Regression - {deg} Degrees)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with linear regression #
predict_years = 6.5

linear_predict = linear_regressor.predict([[predict_years]])
print (linear_predict)

# Predicting a new result with polynomial regression #
polynomial_test = polynomial_regressor.fit_transform([[predict_years]])
polynomial_predict = linear_regressor_2.predict(polynomial_test)
print (polynomial_predict)