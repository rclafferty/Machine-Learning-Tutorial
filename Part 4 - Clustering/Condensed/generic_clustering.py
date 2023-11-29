import core as c

if __name__ == "__main__":
    print("CANNOT RUN AS MAIN FILE -- RUN SPECIFIC CLUSTERING FILE")
    exit()

def run_classification(classifier, name = "NULL Clustering", x_label = "NULL Age", y_label = "NULL Estimated Salary"):
    dataset = c.import_dataset()
    
    x = dataset.iloc[:, -2:].values