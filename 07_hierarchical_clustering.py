# Hierarchical Agglomerative Clustering
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

X, _ = make_blobs(n_samples=100, centers=3, cluster_std=0.8, random_state=42)

Z = linkage(X, method="ward")
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Data point")
plt.ylabel("Distance")
plt.show()

model = AgglomerativeClustering(n_clusters=3, linkage="ward")
labels = model.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis")
plt.title("Agglomerative Hierarchical Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
