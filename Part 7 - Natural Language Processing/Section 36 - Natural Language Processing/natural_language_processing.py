###############################
# NATURAL LANGUAGE PROCESSING #
###############################

# Importing the libraries #
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Importing the dataset #
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

MAX_SIZE = 20000

# Cleaning the texts #
nltk.download('stopwords')
corpus = []
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

for i in dataset.index:
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in all_stopwords]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model #
cv = CountVectorizer(max_features = 1500, min_df = 3)
data_features = cv.fit_transform(corpus).toarray()
dependent_var = dataset.iloc[:, -1].values

# print(len(data_features[0])) # <-- Shows 1,566 words total -- only need most frequent 1,500 words

# Splitting the dataset into the training and test sets #
features_train, features_test, dependent_train, dependent_test = train_test_split(data_features, dependent_var, test_size = 0.2, random_state = 0)

# Training the Naive Bayes model on the training set #
# Naive Bayes does well with NLP, but not the only option
classifier = GaussianNB()
classifier.fit(features_train, dependent_train)

# Predicting the test set results #
predict_test = classifier.predict(features_test) # y_pred

# Making the confusion matrix #
matrix = confusion_matrix(dependent_test, predict_test)
print(matrix)
print(accuracy_score(dependent_test, predict_test))