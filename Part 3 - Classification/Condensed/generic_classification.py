#######################
# LOGISTIC REGRESSION #
#######################

import core as c

if __name__ == "__main__":
    print("CANNOT RUN AS MAIN FILE -- RUN SPECIFIC CLASSIFICATION")
    exit()

def run_classification(classifier, name = "NULL Classification", x_label = "NULL Age", y_label = "NULL Estimated Salary"):
    _ = c.import_dataset()
    x, y = c.split_features_and_dependent()
    x_train, x_test, y_train, y_test = c.split_into_training_and_test_sets(x, y, in_test_size = 0.25)
    x_train, x_test = c.feature_scaling(x_train, x_test)
    
    c.set_classifier(classifier)
    
    c.train_model(x_train, y_train)
    
    # y_pred = c.predict_test_set(x_test)
    # matrix = c.make_confusion_matrix(y_test, y_pred)
    
    c.visualize(x_train, y_train, f"{name} (Training Set)", x_label, y_label)
    c.visualize(x_test, y_test, f"{name} (Test Set)", x_label, y_label)