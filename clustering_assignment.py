import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)

print("Cluster Centers:")
print(kmeans.cluster_centers_)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

# Plot clusters
plt.figure(figsize=(8,6))
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=kmeans.labels_,
    cmap="viridis"
)

plt.title("K-Means Clustering on Iris Dataset (PCA)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.show()
