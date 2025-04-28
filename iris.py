# Import libraries
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

# Load Iris dataset
iris = datasets.load_iris()

# Create DataFrame
dfr = pd.DataFrame(iris.data, columns=iris.feature_names)
dfr['target'] = iris.target

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(dfr[['sepal length (cm)', 'target']])
dfr['kmeans_3'] = kmeans.labels_

# Plot the clusters
plt.scatter(dfr['sepal length (cm)'], dfr['target'], c=dfr['kmeans_3'])
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Target (Species)')
plt.title('KMeans Clustering')
plt.show()

# Evaluate clustering
print("Silhouette Score:", silhouette_score(dfr[['sepal length (cm)', 'target']], dfr['kmeans_3']))
print("Adjusted Rand Index:", adjusted_rand_score(dfr['target'], dfr['kmeans_3']))
print("Normalized Mutual Information:", normalized_mutual_info_score(dfr['target'], dfr['kmeans_3']))
