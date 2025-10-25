from sklearn.datasets import load_breast_cancer, load_wine

data = load_breast_cancer()
X = data.data
y = data.target

for i in range(len(data.data)):
    print(f"{i}: {data.data[i]} -> {data.target_names[data.target[i]]}")