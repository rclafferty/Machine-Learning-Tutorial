###########
# APRIORI #
###########

# Importing the libraries #
import os
import pandas as pd
from apyori import apriori

current_file = None
# Importing the dataset #
for _, _, files in os.walk('.'):
    for file in files:
        if file.endswith('.csv'):
            current_file = file
            break
        
dataset = pd.read_csv(current_file, header = None)
transactions_list = []

for row in dataset.iloc:
    transactions_list.append([str(col) for col in row])

# Training the Apriori model on the dataset #
rules = apriori(
    transactions = transactions_list,
    min_support = 0.003,
    min_confidence = 0.2, # 20% correlation(?)
    min_lift = 3,
    min_length = 2, # Only 2 products in the rule
    max_length = 2
)

# Visualising the results #
# Displaying the first results coming directly from the output of the apriori function
results = list(rules)

def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts = [result[2][0][3] for result in results]
    
    return list(zip(lhs, rhs, supports, confidences, lifts))

results_in_dataframe = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

n_largest = results_in_dataframe.nlargest(n = 10, columns = 'Lift')
