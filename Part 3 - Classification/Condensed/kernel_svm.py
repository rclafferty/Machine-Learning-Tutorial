##############
# KERNEL SVM #
##############

import generic_classification
from sklearn.svm import SVC

classifier = SVC(kernel = 'rbf', random_state = 0)
generic_classification.run_classification(classifier, "Kernel SVM", "Age", "Estimated Salary")