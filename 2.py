from sklearn.datasets import load_breast_cancer, load_wine, load_iris

data = load_iris()
X = data.data
y = data.target

for i in range(len(data.data)):
    print(f"{i}: {data.data[i]} -> {data.target_names[data.target[i]]}")