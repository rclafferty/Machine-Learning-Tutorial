import os
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


dataset = None
classifier = None
ss = StandardScaler()


def set_classifier(in_classifier):
    global classifier
    classifier = in_classifier


def import_dataset(in_filename = None):
    global dataset

    if in_filename is None:
        # Importing the dataset #
        for _, _, files in os.walk('.'):
            for file in files:
                if file.endswith('.csv'):
                    in_filename = file
                    break

    dataset = pd.read_csv(in_filename)

    return dataset


def split_features_and_dependent(dependent_columns = 1):
    global dataset
    x = dataset.iloc[:, :-dependent_columns].values
    y = dataset.iloc[:, -dependent_columns].values

    return x, y


def split_into_training_and_test_sets(x, y, in_test_size=0.2):
    global dataset

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=in_test_size, random_state=1)

    return x_train, x_test, y_train, y_test


def feature_scaling(x_train, x_test):
    global ss

    x_train_fit = ss.fit_transform(x_train)
    x_test_fit = ss.transform(x_test)

    return x_train_fit, x_test_fit


def train_model(x_train, y_train):
    global classifier

    classifier.fit(x_train, y_train)


def predict_test_set(x_test):
    global classifier

    predict_test = classifier.predict(x_test)

    return predict_test


def predict_value(values=[]):
    global ss
    global classifier

    scaled_predict_x = ss.transform([values])
    prediction = classifier.predict(scaled_predict_x)

    return prediction


def make_confusion_matrix(y_test, predict_test):
    return confusion_matrix(y_test, predict_test)


def visualize(x, y, title, x_label, y_label, colors=['red', 'green']):
    X_set, y_set = ss.inverse_transform(x), y
    X1, X2 = np.meshgrid(
        np.arange(start=X_set[:, 0].min() - 10,
                  stop=X_set[:, 0].max() + 10, step=0.25),
        np.arange(start=X_set[:, 1].min() - 1000,
                  stop=X_set[:, 1].max() + 1000, step=0.25)
    )
    plt.contourf(X1, X2, classifier.predict(ss.transform(np.array([X1.ravel(
    ), X2.ravel()]).T)).reshape(X1.shape), alpha=0.75, cmap=ListedColormap(colors))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c=ListedColormap(colors)(i), label=j)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()
