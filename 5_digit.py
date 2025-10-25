from sklearn.datasets import fetch_openml

# Numer interesującej nas cyfry w zbiorze
num = 2

print("Wczytuję dane MNIST...")
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.astype('float32')
Y = mnist.target.astype('int')

# Weźmy interesującą nas cyfrę
digit = X.iloc[num].values

# Przekształćmy ja do formatu macierzy 28x28
digit_image = digit.reshape(28, 28)

# Ustawiamy próg konwersji na binarne wartości (1 i 0)
binary_image = (digit_image > 127/255).astype(int)

# Wyświetlenie cyfry w postaci binarnej (0 i 1)
print(f"Cyfra {Y.iloc[num]}")
print("\nReprezentacja binarna (0 i 1):")
for row in binary_image:
    print(' '.join(map(str, row)))

# Wyświetlenie cyfry w bardziej czytelnej formie
print("\nWizualizacja w terminalu:")
for row in binary_image:
    line = ''
    for pixel in row:
        if pixel == 1:
            line += '#'
        else:
            line += ' '
    print(line)