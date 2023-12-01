##########################
# UPPER CONFIDENCE BOUND #
##########################

# Importing the libraries #
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

current_file = None
# Importing the dataset #
for _, _, files in os.walk('.'):
    for file in files:
        if file.endswith('.csv'):
            current_file = file
            break
        
dataset = pd.read_csv(current_file)

N = 10000 # Total number of users
SUPER_HIGH_VALUE = N + 10 # Tutorial uses 1e400, but this uses less space
d = 10 # Number of ads
ads_selected = [] # Full list of ads selected over the rounds
numbers_of_selections = [0] * d # Number of times ads i was selected (Starts as list of 10 0's)
sums_of_rewards = [0] * d # Sum of rewards of ad i (Starts as list of 10 0's)
total_reward = 0 # Running total of rewards

for round in range(N):
    ad = 0
    max_upper_bound = 0
    
    for i in range(d):
        upper_bound = 0
        
        if numbers_of_selections[i] == 0:
            upper_bound = SUPER_HIGH_VALUE
        else:
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt((3/2) * (math.log(round + 1) / numbers_of_selections[i]))
            
            upper_bound = average_reward + delta_i
            
        if upper_bound > max_upper_bound:
            ad = i
            max_upper_bound = upper_bound
            
    ads_selected.append(ad)
    numbers_of_selections[ad] += 1
    reward = dataset.values[round, ad]
    sums_of_rewards[ad] += reward
    total_reward += reward
    
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Numbers of times ad was selected')
plt.show()