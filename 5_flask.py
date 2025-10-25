import joblib
import numpy as np
from flask import Flask, render_template, request
from PIL import Image

app = Flask(__name__)
model = joblib.load('mnist_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def digit():
    prediction = None
    if request.method == 'POST':
        # Pobranie pliku
        file = request.files['image']

        # Konwersja pliku graficznego na skale szarości
        img = Image.open(file).convert('L')
        # Zmiana rozmiaru na 28x28
        img = img.resize((28, 28))
        # Zmiana pliku graficznego na tablicę
        img_array = np.array(img)
        # Stworzenie wektora
        img_vector = (255 - img_array).reshape(1, -1) / 255

        # Predykcja - czyli zgadywanie co jest na obrazie
        prediction = model.predict(img_vector)[0]

    return render_template("digit.html", prediction=str(prediction))


if __name__ == '__main__':
    app.run(debug=True)