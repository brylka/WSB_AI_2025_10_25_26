from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree

iris = load_iris()
X = iris.data
y = iris.target

# for i in range(len(iris.data)):
#     print(f"{i}: {iris.data[i]} -> {iris.target_names[iris.target[i]]}")

#                                podział zbiory (ilość danych treningowych) \/
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dt_classifier = DecisionTreeClassifier(
    criterion='gini',       # 'entropy'
    max_depth=3,            # głębokość
    min_samples_split=2,    # minimalna ilość próbek w węźle
    random_state=42
)

dt_classifier.fit(X_train, y_train)

y_pred = dt_classifier.predict(X_test)

print(f"Dokładność: {accuracy_score(y_test, y_pred)}")
print("Raport klasyfikacji:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

plt.figure(figsize=(15,10))
tree.plot_tree(dt_classifier, feature_names=iris.feature_names,
               class_names=iris.target_names, filled=True)
plt.show()