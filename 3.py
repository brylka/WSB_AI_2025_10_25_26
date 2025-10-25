import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

k_range = range(1, 31)
scores = []

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    scores.append(knn.score(X_test_scaled, y_test))

plt.figure(figsize=(10,6))
plt.plot(k_range, scores, marker='o')
plt.xlabel('Wartość k')
plt.ylabel('Dokładność')
plt.title('Dokładność KNN dla różnych wartości k')
plt.grid(True)
plt.show()

optimal_k = k_range[np.argmax(scores)]
print(f"Optymalne k: {optimal_k}")

knn_classifier = KNeighborsClassifier(
    n_neighbors=optimal_k,
    weights='distance', # uniform
    metric='euclidean'
)

knn_classifier.fit(X_train_scaled, y_train)

y_pred_knn = knn_classifier.predict(X_test_scaled)
print(f"Dokładność KNN: {accuracy_score(y_test, y_pred_knn)}")
print("Macież pomyłek")
print(confusion_matrix(y_test, y_pred_knn))