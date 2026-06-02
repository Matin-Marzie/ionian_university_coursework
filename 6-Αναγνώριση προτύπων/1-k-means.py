import numpy as np

def KMeans(D, k, e, max_iters=100):
    # Initialization of centroids
    indices = np.random.choice(D.shape[0], k, replace=False)
    centroids = D[indices]

    old_centroids = np.zeros((k, D.shape[1]))


    for _ in range(max_iters):
        clusters = [[] for _ in range(k)]

        # Ανάθεση κάθε αντικειμένου σε μια συστάδα
        for point in D:
            distances = np.linalg.norm(point - centroids, axis=1)
            cluster_index = np.argmin(distances)
            clusters[cluster_index].append(point)

        # Ανανέωση του κέντρου της συστάδας
        old_centroids = centroids.copy()

        for i in range(len(clusters)):
            if len(clusters[i]) != 0:
                centroids[i] = np.mean(np.array(clusters[i]), axis=0)

        if np.linalg.norm(centroids - old_centroids) < e:
            break

    return centroids, clusters





from sklearn import datasets

iris_samples = datasets.load_iris()

# Keeping only two characteristics
X = iris_samples.data[:, :2]
Y = iris_samples.target

# Βρίσκουμε τις διακρητές πραγματικές ομάδες
k=len(np.unique(Y))
centroids, clusters = KMeans(X, k, e=1e-4)





import matplotlib.pyplot as plt

colors = ['red', 'green', 'blue']

# One row with two columns
figure, axes = plt.subplots(1, 2, figsize=(12, 6))

# scatters
for point, label in zip(X, Y):
    axes[0].scatter(point[0], point[1], color=colors[label])

# Legend    
for i, name in enumerate(iris_samples.target_names):
    axes[0].scatter([], [], color=colors[i], label=name)

axes[0].set_xlabel('Sepal length')
axes[0].set_ylabel('Sepal width')
axes[0].set_title(f"Original Dataset")
axes[0].legend(loc='best')

# Centroids
for i, centroid in enumerate(centroids):
    # centroid  # Legend
    axes[1].scatter(centroid[0], centroid[1], color=colors[i], s=200, label=f"cluster {i}")

# Clusters
for index, cluster in enumerate(clusters):
    for point in cluster:
        axes[1].scatter(point[0], point[1], color=colors[index])



axes[1].set_xlabel('Sepal length')
axes[1].set_ylabel('Sepal width')
axes[1].set_title(f"KMeans (k={k})")
axes[1].legend(loc='best')

plt.tight_layout()  # Adjust layout to avoid overlap
plt.show()  