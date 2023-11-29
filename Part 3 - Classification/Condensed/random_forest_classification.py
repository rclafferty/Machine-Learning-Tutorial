################################
# RANDOM FOREST CLASSIFICATION #
################################

import generic_classification
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
generic_classification.run_classification(classifier, "Random Forest", "Age", "Estimated Salary")