import joblib
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print("Wczytuję dane MNIST...")
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.astype('float32')
y = mnist.target.astype('int')

# Normalizacja danych (skalowanie do przedziału [0,1])
X = X / 255

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=200)

print("Trenuję model...")
rf.fit(X_train, y_train)

print("Zapisuję model do pliku...")
joblib.dump(rf, 'mnist_model.pkl')