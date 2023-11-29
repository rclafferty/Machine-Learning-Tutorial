##########################
# SUPPORT VECTOR MACHINE #
##########################

import generic_classification
from sklearn.svm import SVC

classifier = SVC(kernel = 'linear', random_state = 0)
generic_classification.run_classification(classifier, "SVM", "Age", "Estimated Salary")