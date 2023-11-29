#######################
# LOGISTIC REGRESSION #
#######################

import generic_classification
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)
generic_classification.run_classification(classifier, "Logistic Regression", "Age", "Estimated Salary")