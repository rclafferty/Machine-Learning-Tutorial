###########################
# HIERARCHICAL CLUSTERING #
###########################

# Importing the libraries #
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

current_file = None
# Importing the dataset #
for _, _, files in os.walk('.'):
    for file in files:
        if file.endswith('.csv'):
            current_file = file
            break
        
dataset = pd.read_csv(current_file)

# ALL columns are features!
# Remove all but two columns so we can keep it to two dimensions --> Income + Score
data_features = dataset.iloc[:, -2:].values 

# Using the dendrogram to find the optimal number of clusters #
dendrogram = sch.dendrogram(sch.linkage(data_features, method = 'ward'))

plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()

# Training the Hierarchical Clustering model on the dataset #
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(data_features)

# Visualising the clusters
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
for i in range(0, 5):
    plt.scatter(
        data_features[y_hc == i, 0],
        data_features[y_hc == i, 1],
        s = 100,
        c = colors[i],
        label = f'Cluster {i}'
    )
plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.legend()
plt.show()