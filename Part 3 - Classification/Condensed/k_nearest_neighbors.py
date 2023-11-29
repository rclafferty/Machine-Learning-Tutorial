#######################
# K-NEAREST NEIGHBORS #
#######################

import generic_classification
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()
generic_classification.run_classification(classifier, "K-Nearest Neighbors", "Age", "Estimated Salary")