#####################
# THOMPSON SAMPLING #
#####################

# Importing the libraries #
import os
import matplotlib.pyplot as plt
import pandas as pd
import random

current_file = None
# Importing the dataset #
for _, _, files in os.walk('.'):
    for file in files:
        if file.endswith('.csv'):
            current_file = file
            break
        
dataset = pd.read_csv(current_file)

N = 500 # Total number of users
SUPER_HIGH_VALUE = N + 10 # Tutorial uses 1e400, but this uses less space
d = 10 # Number of ads
ads_selected = [] # Full list of ads selected over the rounds
numbers_of_rewards_0 = [0] * d
numbers_of_rewards_1 = [0] * d
total_reward = 0 # Running total of rewards

for round in range(N):
    ad = 0
    max_random = 0
    
    for i in range(d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        
        if random_beta > max_random:
            max_random = random_beta
            ad = i
            
    ads_selected.append(ad)
    reward = dataset.values[round, ad]
    if reward == 1:
        numbers_of_rewards_1[ad] += 1
    else:
        numbers_of_rewards_0[ad] += 1
        
    total_reward += reward
    
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Numbers of times ad was selected')
plt.show()