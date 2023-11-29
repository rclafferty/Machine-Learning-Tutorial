######################
# K-MEANS CLUSTERING #
######################

# Importing the libraries #
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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

# Use the elbow method to find the optimal number of clusters #
wcss = []
for i in range(1, 11): # 1-10, 11 is excluded
    kmeans = KMeans(n_clusters = i, random_state = 42) # init = 'k-means++'
    kmeans.fit(data_features)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Training the K-Means model on the dataset #
kmeans = KMeans(n_clusters = 5, random_state = 42) # init = 'k-means++'
y_kmeans = kmeans.fit_predict(data_features)

# Visualising the clusters
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
for i in range(0, 5):
    plt.scatter(
        data_features[y_kmeans == i, 0],
        data_features[y_kmeans == i, 1],
        s = 100,
        c = colors[i],
        label = f'Cluster {i}'
    )
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s = 300,
    c = 'yellow',
    label = 'Centroids'
)
plt.title('Clusters of Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.legend()
plt.show()