############################
# DECISION TREE REGRESSION #
############################

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

# Training the decision tree regression model on the whole dataset #

# Predicting a new result #

# Visualizing the decision tree regression results #