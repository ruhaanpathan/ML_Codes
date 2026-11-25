# DBSCAN Clustering
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

X, _ = make_moons(n_samples=300, noise=0.08, random_state=42)
X = StandardScaler().fit_transform(X)

model = DBSCAN(eps=0.25, min_samples=5)
labels = model.fit_predict(X)

print("Cluster labels:", set(labels))
print("Number of clusters (excluding noise):", len(set(labels)) - (1 if -1 in labels else 0))
print("Noise points:", list(labels).count(-1))

valid = labels != -1
if len(set(labels[valid])) > 1:
    print("Silhouette score (excluding noise):", silhouette_score(X[valid], labels[valid]))

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=30)
plt.title("DBSCAN Clustering (-1 represents noise)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
