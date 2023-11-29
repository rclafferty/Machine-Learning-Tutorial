###############
# NAIVE BAYES #
###############

import generic_classification
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
generic_classification.run_classification(classifier, "Naive Bayes", "Age", "Estimated Salary")