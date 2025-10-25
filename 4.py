import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
import matplotlib.pyplot as plt

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gnb_classifier = GaussianNB()
gnb_classifier.fit(X_train, y_train)

y_pred_gnb = gnb_classifier.predict(X_test)
y_proba_gnb = gnb_classifier.predict_proba(X_test)

print(f"Dokładność GNB: {accuracy_score(y_test, y_pred_gnb)}")
print(f"Raport klasyfikacji:")
print(classification_report(y_test, y_pred_gnb, target_names=data.target_names))

y_test_bin = label_binarize(y_test, classes=[0,1,2])
n_classes = y_test_bin.shape[1]

plt.figure(figsize=(10,8))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:,i], y_proba_gnb[:,i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{data.target_names[i]} (AUC = {roc_auc}")

plt.plot([0,1], [0,1], 'k--', label="Losowy klasyfikator")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Krzywa ROC - Naive Bayes')
plt.legend()
plt.grid(True)
plt.show()