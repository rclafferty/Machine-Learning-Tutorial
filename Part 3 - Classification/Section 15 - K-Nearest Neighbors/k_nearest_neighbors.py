#######################
# K-NEAREST NEIGHBORS #
#######################

# Importing the libraries #
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

current_file = None
# Importing the dataset #
for _, _, files in os.walk('.'):
    for file in files:
        if file.endswith('.csv'):
            current_file = file
            break
        
dataset = pd.read_csv(current_file)

data_features = dataset.iloc[:, :-1].values # Take all columns EXCEPT last one
dependent_var = dataset.iloc[:, -1].values # Take ONLY last column

# Splitting the dataset into the training set and testing set #
# 80% in training set, 20% in test set
features_train, features_test, dependent_train, dependent_test = train_test_split(data_features, dependent_var, test_size = 0.25, random_state = 1)

# Feature scaling - Standardization #
ss = StandardScaler()
features_train = ss.fit_transform(features_train)
features_test = ss.transform(features_test)

# Training the K-Nearest Neighbors model on the training set #
classifier = KNeighborsClassifier() # n_neighbors = 5, metric = 'minkowski', p = 2
classifier.fit(features_train, dependent_train)

predict_age = 30
predict_salary = 87000

scaled_predict_features = ss.transform([[predict_age, predict_salary], [30, 100000]])
predict_purchased = classifier.predict(scaled_predict_features)

# Predicting the test set results #
predict_test = classifier.predict(features_test) # y_pred

np.set_printoptions(precision=2)
# print(np.concatenate((predict_test.reshape(len(predict_test), 1), dependent_test.reshape(len(dependent_test), 1)), axis=1))

# Making the confusion matrix #
matrix = confusion_matrix(dependent_test, predict_test)
print(matrix)

print(accuracy_score(dependent_test, predict_test))

# Visualizing the training set results #
from matplotlib.colors import ListedColormap
h = .02
X0_min, X0_max = features_train[:, 0].min() - 1, features_train[:, 0].max() + 1
X1_min, X1_max = features_train[:, 1].min() - 1, features_train[:, 1].max() + 1
X0, X1 = np.meshgrid(np.arange(X0_min, X0_max, h), np.arange(X1_min, X1_max, h))
X0_unscaled, X1_unscaled = ss.inverse_transform(np.c_[X0.ravel(), X1.ravel()]).T.reshape(2, X0.shape[0], X0.shape[1])
colormap = ListedColormap(['salmon', 'dodgerblue'])
plt.contourf(X0_unscaled, X1_unscaled, classifier.predict(np.c_[X0.ravel(), X1.ravel()]).reshape(X0.shape),
             alpha=0.75, cmap=colormap)
X_set, y_set = ss.inverse_transform(features_train), dependent_train
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = colormap(i), label = j)
xLower, xUpper = min(X_set[:, 0]), max(X_set[:, 0])
yLower, yUpper = min(X_set[:, 1]), max(X_set[:, 1])
xMargin, yMargin = xUpper * 0.05, yUpper * 0.05
plt.xlim(xLower - xMargin, xUpper + xMargin)
plt.ylim(yLower - yMargin, yUpper + yMargin)
plt.title("K-NN (Training set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
h = .02
X0_min, X0_max = features_test[:, 0].min() - 1, features_test[:, 0].max() + 1
X1_min, X1_max = features_test[:, 1].min() - 1, features_test[:, 1].max() + 1
X0, X1 = np.meshgrid(np.arange(X0_min, X0_max, h), np.arange(X1_min, X1_max, h))
X0_unscaled, X1_unscaled = ss.inverse_transform(np.c_[X0.ravel(), X1.ravel()]).T.reshape(2, X0.shape[0], X0.shape[1])
colormap = ListedColormap(['salmon', 'dodgerblue'])
plt.contourf(X0_unscaled, X1_unscaled, classifier.predict(np.c_[X0.ravel(), X1.ravel()]).reshape(X0.shape),
             alpha=0.75, cmap=colormap)
X_set, y_set = ss.inverse_transform(features_test), dependent_test
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = colormap(i), label = j)
xLower, xUpper = min(X_set[:, 0]), max(X_set[:, 0])
yLower, yUpper = min(X_set[:, 1]), max(X_set[:, 1])
xMargin, yMargin = xUpper * 0.05, yUpper * 0.05
plt.xlim(xLower - xMargin, xUpper + xMargin)
plt.ylim(yLower - yMargin, yUpper + yMargin)
plt.title("K-NN (Test set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()

