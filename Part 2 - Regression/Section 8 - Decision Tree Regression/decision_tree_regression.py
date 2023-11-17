############################
# DECISION TREE REGRESSION #
############################

# Importing the libraries #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# Importing the dataset #
dataset = pd.read_csv('Position_Salaries.csv')

data_features = dataset.iloc[:, 1:-1].values # Ignore the first column (extra -- just labels)
dependent_var = dataset.iloc[:, -1].values

# Training the decision tree regression model on the whole dataset #
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(data_features, dependent_var)

# Predicting a new result #
predict_years = 6.5

prediction = regressor.predict([[predict_years]])

# Visualizing the decision tree regression results #

features_grid = np.arange(min(data_features), max(data_features), 0.1)
features_grid = features_grid.reshape((len(features_grid), 1))

plt.scatter(data_features, dependent_var, color = 'red') # real results
plt.plot(features_grid, regressor.predict(features_grid), color = 'blue') # predicted results
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()