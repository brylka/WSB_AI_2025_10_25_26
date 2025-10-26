from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt

# Generowanie danych o nieregularnym kształcie
X_moons, y_moons = make_moons(n_samples=300, noise=0.05, random_state=42)

# Standaryzacja (ważna dla DBSCAN!)
scaler = StandardScaler()
X_moons_scaled = scaler.fit_transform(X_moons)

# Znalezienie optymalnego eps - metoda k-distance
neighbors = NearestNeighbors(n_neighbors=5)
neighbors.fit(X_moons_scaled)
distances, indices = neighbors.kneighbors(X_moons_scaled)

# Sortowanie odległości
distances = np.sort(distances[:, -1], axis=0)

plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.xlabel('Punkty posortowane według odległości')
plt.ylabel('5-ta najbliższa odległość')
plt.title('Wykres k-distance dla wyboru eps')
plt.grid(True)
plt.show()

# DBSCAN z wybranymi parametrami
dbscan = DBSCAN(
    eps=0.24,           # promień sąsiedztwa
    min_samples=5,     # minimalna liczba punktów
    metric='euclidean'
)
y_dbscan = dbscan.fit_predict(X_moons_scaled)

# Liczba klastrów i outlierów
n_clusters = len(set(y_dbscan)) - (1 if -1 in y_dbscan else 0)
n_noise = list(y_dbscan).count(-1)

print(f"Liczba klastrów: {n_clusters}")
print(f"Liczba outlierów: {n_noise}")

# Wizualizacja wyników
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Prawdziwe etykiety
axes[0].scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap='viridis', alpha=0.6)
axes[0].set_title('Prawdziwe etykiety')
axes[0].set_xlabel('Cecha 1')
axes[0].set_ylabel('Cecha 2')

# K-means (dla porównania)
from sklearn.cluster import KMeans
kmeans_moons = KMeans(n_clusters=2, random_state=42)
y_kmeans_moons = kmeans_moons.fit_predict(X_moons)
axes[1].scatter(X_moons[:, 0], X_moons[:, 1], c=y_kmeans_moons, cmap='viridis', alpha=0.6)
axes[1].scatter(kmeans_moons.cluster_centers_[:, 0], kmeans_moons.cluster_centers_[:, 1],
                s=300, c='red', marker='X', edgecolors='black', linewidths=2)
axes[1].set_title('K-means (nie radzi sobie)')
axes[1].set_xlabel('Cecha 1')
axes[1].set_ylabel('Cecha 2')

# DBSCAN
axes[2].scatter(X_moons[:, 0], X_moons[:, 1], c=y_dbscan, cmap='viridis', alpha=0.6)
# Oznacz outliery tylko jeśli istnieją
outliers = X_moons[y_dbscan == -1]
if len(outliers) > 0:
    axes[2].scatter(outliers[:, 0], outliers[:, 1], c='red', marker='x',
                    s=100, linewidths=2, label='Outliers')
    axes[2].legend()  # Legenda tylko gdy są outliers
axes[2].set_title(f'DBSCAN (radzi sobie dobrze) - {n_noise} outliers')
axes[2].set_xlabel('Cecha 1')
axes[2].set_ylabel('Cecha 2')

plt.tight_layout()
plt.show()
