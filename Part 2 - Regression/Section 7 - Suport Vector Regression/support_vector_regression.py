#############################
# SUPPORT VECTOR REGRESSION #
#############################

# Importing the libraries #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Importing the dataset #
dataset = pd.read_csv('Position_Salaries.csv')

data_features = dataset.iloc[:, 1:-1].values # Ignore the first column (extra -- just labels)
dependent_var = dataset.iloc[:, -1].values

# Feature Scaling #
dependent_var = dependent_var.reshape(len(dependent_var), 1)

# Avoid some features being dominated by other features and imbalance the model using standardization
ss_features = StandardScaler()
data_features = ss_features.fit_transform(data_features)

ss_dependent = StandardScaler()
dependent_var = ss_dependent.fit_transform(dependent_var)

# Training the SVR model on the whole dataset #
regressor = SVR(kernel = 'rbf')
regressor.fit(data_features, dependent_var)

# Predicting a new result #
predict_years = 6.5

ss_features_transform = ss_features.transform([[predict_years]])
regressor_prediction = regressor.predict(ss_features_transform)
ss_prediction = ss_dependent.inverse_transform(regressor_prediction.reshape(-1, 1))

print (ss_prediction)
print (data_features)
print (dependent_var)

features_inverse = ss_features.inverse_transform(data_features)
dependent_inverse = ss_dependent.inverse_transform(dependent_var)

# Visualizing the SVR results #
plt.scatter(features_inverse, dependent_inverse, color = 'red') # real results
plt.plot(features_inverse, ss_dependent.inverse_transform(regressor.predict(data_features).reshape(-1, 1)), color = 'blue') # predicted results
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualizing the SVR results (for higher resolution and smoother curve) #
features_grid = np.arange(min(features_inverse), max(features_inverse), 0.1)
features_grid = features_grid.reshape((len(features_grid), 1))

ss_features_transform_hr = ss_features.transform(features_grid)
regressor_prediction_hr = regressor.predict(ss_features_transform_hr)
ss_prediction_hr = ss_dependent.inverse_transform(regressor_prediction_hr.reshape(-1, 1))

plt.scatter(features_inverse, dependent_inverse, color = 'red') # real results
plt.plot(features_grid, ss_prediction_hr, color = 'blue') # predicted results
plt.title('Truth or Bluff (SVR - High Resolution)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
