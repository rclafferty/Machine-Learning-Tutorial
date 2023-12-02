#############################
# ARTIFICIAL NEURAL NETWORK #
#############################

### PART 1: Data Preprocessing ###

# Importing the libraries #
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

current_file = None
# Importing the dataset #
for _, _, files in os.walk('.'):
    for file in files:
        if file.endswith('.csv'):
            current_file = file
            break
        
dataset = pd.read_csv(current_file)

data_features = dataset.iloc[:, 3:-1].values # Take all columns EXCEPT last one
dependent_var = dataset.iloc[:, -1].values # Take ONLY last column

# Encoding categorical data #

# Label encoding the "gender" column
le = LabelEncoder()
data_features[:, 2] = le.fit_transform(data_features[:, 2])

# One Hot Encoding the "geography" column
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
data_features = ct.fit_transform(data_features)

# Splitting the dataset into the training set and testing set #
# 80% in training set, 20% in test set
features_train, features_test, dependent_train, dependent_test = train_test_split(data_features, dependent_var, test_size = 0.2, random_state = 0)

# Feature scaling #
ss = StandardScaler()
features_train = ss.fit_transform(features_train)
features_test = ss.fit_transform(features_test)

### PART 2: Building the ANN ###
INPUT_NEURONS = 7
FIRST_HIDDEN_LAYER_NEURONS = 6
SECOND_HIDDEN_LAYER_NEURONS = 6
OUTPUT_NEURONS = 1

# Initializing the ANN
ann = Sequential()

# Adding the input layer and the first hidden layer
ann.add(Dense(units = FIRST_HIDDEN_LAYER_NEURONS, activation = 'relu'))

# Adding the second hidden layer
ann.add(Dense(units = SECOND_HIDDEN_LAYER_NEURONS, activation = 'relu'))

# Adding the output layer
ann.add(Dense(units = OUTPUT_NEURONS, activation = 'sigmoid'))

### PART 3: Training the ANN ###
TRAINING_BATCH_SIZE = 32
TRAINING_EPOCHS = 100

# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN on the training set
ann.fit(features_train, dependent_train, batch_size = TRAINING_BATCH_SIZE, epochs = TRAINING_EPOCHS)

### PART 4: Making the predictions and evaluating the model ###

# Predicting the result of a single observation
PREDICT_GEOGRAPHY = 'France'
PREDICT_CREDIT_SCORE = 600
PREDICT_GENDER = 1 # Encoded 1 = Male, 0 = Female
PREDICT_AGE = 40
PREDICT_TENURE = 3
PREDICT_BALANCE = 60000
PREDICT_PRODUCTS = 2
PREDICT_CC = 1 # Encoded 1 = Yes, 2 = No
PREDICT_ACTIVE = 1 # Encoded 1 = Yes, 2 = No
PREDICT_SALARY = 50000

features_predict = [
    # PREDICT_GEOGRAPHY, # France needs to be encoded
    1, 0, 0, # Encoding of France
    PREDICT_CREDIT_SCORE,
    PREDICT_GENDER, # Encoded 1 = Male, 0 = Female
    PREDICT_AGE,
    PREDICT_TENURE,
    PREDICT_BALANCE,
    PREDICT_PRODUCTS,
    PREDICT_CC, # Encoded 1 = Yes, 2 = No
    PREDICT_ACTIVE, # Encoded 1 = Yes, 2 = No
    PREDICT_SALARY
]

prediction = ann.predict(ss.transform([features_predict])) # Probability of leaving
print (f'Probability of leaving: {prediction}') # 0.038
print (f'Leaving ? {prediction > 0.5}') # 0 or False

# Predicting the test set results
predict_test = ann.predict(features_test)
predict_test = (predict_test > 0.5)
print(np.concatenate((predict_test.reshape(len(predict_test), 1), dependent_test.reshape(len(dependent_test), 1)), axis=1))

# Making the confusion matrix
matrix = confusion_matrix(dependent_test, predict_test)
print(matrix)
print(accuracy_score(dependent_test, predict_test))
