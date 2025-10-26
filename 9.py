from sklearn.datasets import make_circles, make_blobs, make_moons
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

datasets = [
    make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=42),
    make_moons(n_samples=300, noise=0.05, random_state=42),
    make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)
]

dataset_names = ['Kulowe klastry', 'Klastry księżycowe', 'Klastry okrągłe']

fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for i, (X, y) in enumerate(datasets):
    # Standaryzacja
    X_scaled = StandardScaler().fit_transform(X)

    # Oryginalne dane
    axes[i, 0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.6)
    axes[i, 0].set_title(f'{dataset_names[i]} - Oryginał')
    axes[i, 0].set_ylabel('Cecha 2')

    # K-means
    kmeans = KMeans(n_clusters=2, random_state=42)
    y_kmeans = kmeans.fit_predict(X)
    axes[i, 1].scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', alpha=0.6)
    axes[i, 1].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                       s=200, c='red', marker='X', edgecolors='black', linewidths=2)
    axes[i, 1].set_title(f'K-means (Silhouette: {silhouette_score(X, y_kmeans):.3f})')

    # DBSCAN
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    y_dbscan = dbscan.fit_predict(X_scaled)
    axes[i, 2].scatter(X[:, 0], X[:, 1], c=y_dbscan, cmap='viridis', alpha=0.6)
    outliers = X[y_dbscan == -1]
    if len(outliers) > 0:
        axes[i, 2].scatter(outliers[:, 0], outliers[:, 1], c='red',
                           marker='x', s=100, linewidths=2)

    # Oblicz silhouette tylko dla punktów nie będących outlierami
    mask = y_dbscan != -1
    if mask.sum() > 0 and len(set(y_dbscan[mask])) > 1:
        sil_score = silhouette_score(X[mask], y_dbscan[mask])
    else:
        sil_score = 0
    axes[i, 2].set_title(f'DBSCAN (Silhouette: {sil_score:.3f})')

for ax in axes[-1]:
    ax.set_xlabel('Cecha 1')

plt.tight_layout()
plt.show()
