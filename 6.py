import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

X_blobs, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

inertias = []
silhouette_scores = []
K_range = range(2,11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_blobs)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_blobs, kmeans.labels_))

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,5))

ax1.plot(K_range, inertias, 'bo-')
ax1.set_xlabel('Liczba klastrów (k)')
ax2.set_ylabel('Inercja (suma kwadratów odległości)')
ax1.set_title('Metoda łokcia')
ax1.grid(True)

ax2.plot(K_range, silhouette_scores, 'ro-')
ax2.set_xlabel('Liczba klastrów (k)')
ax2.set_ylabel('Współczynnik sylwetkowy')
ax2.set_title('Współczynnik sylwetkowy dla różnych k')
ax2.grid(True)

plt.tight_layout()
plt.show()

