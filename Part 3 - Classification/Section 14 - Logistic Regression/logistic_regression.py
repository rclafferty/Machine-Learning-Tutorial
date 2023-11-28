#######################
# LOGISTIC REGRESSION #
#######################

# Importing the libraries #
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
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

# Training the Logistic Regression model on the training set #
classifier = LogisticRegression(random_state = 0)
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
X_set, y_set = ss.inverse_transform(features_train), dependent_train
X1, X2 = np.meshgrid(
    np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
    np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25)
)
plt.contourf(X1, X2, classifier.predict(ss.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(['red', 'green']))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(['red', 'green'])(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = ss.inverse_transform(features_test), dependent_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(ss.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(['red', 'green']))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(['red', 'green'])(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

