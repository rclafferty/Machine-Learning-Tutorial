################################
# DECISION TREE CLASSIFICATION #
################################

import generic_classification
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion = 'entropy')
generic_classification.run_classification(classifier, "Decision Tree", "Age", "Estimated Salary")