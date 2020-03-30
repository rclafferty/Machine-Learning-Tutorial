# Polynomial Regression
# Predict a salary at a certain point in the heirarchy

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values # Levels
y = dataset.iloc[:, 2:3].values # Salary

# Splitting the dataset into the Training set and Test set
# Need ALL steps, so no train/test sets -- Need max info possible
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
# Uses similar steps to Multiple Linear Regression -- No feature scaling necessary
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Linear Regression to the dataset
# Only used for comparing to polynomial regression
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
polynomial_regressor = PolynomialFeatures(degree = 4)
X_poly = polynomial_regressor.fit_transform(X) # X_poly[:, 0] is x^0, X_poly[:, 1] is x^1, X_poly[:, 2] is x^2
linear_regressor2 = LinearRegression()
linear_regressor2.fit(X_poly, y) # Fit X_poly (polynomial regression fit) to the model

# Visualizing the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, linear_regressor.predict(X), color = 'blue')
plt.title('Salary Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualizing the Polynomial Regression results
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, linear_regressor2.predict(polynomial_regressor.fit_transform(X_grid)), color = 'blue')
plt.title('Salary Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
linear_regressor.predict([[6.5]])

# Predicting a new result with Polynomial Regression
linear_regressor2.predict(polynomial_regressor.fit_transform([[6.5]])) #158k -- Close enough to 160k --> TRUTH!