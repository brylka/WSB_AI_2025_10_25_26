from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. WCZYTANIE DANYCH
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. DEFINICJA PARAMETRÓW
param_grid = {
    'max_depth': [2, 3, 4, 5, 6],
    'min_samples_split': [2, 5, 10, 15]
}

# 3. GRIDSEARCHCV
model = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    return_train_score=True  # ważne dla wizualizacji!
)

print("Przeszukuję parametry...")
grid_search.fit(X_train, y_train)

# 4. PRZYGOTOWANIE DANYCH DO WIZUALIZACJI
results = pd.DataFrame(grid_search.cv_results_)

# ==============================================================================
# WIZUALIZACJA 1: HEATMAPA WYNIKÓW
# ==============================================================================
plt.figure(figsize=(10, 6))

# Pivot table dla heatmapy
pivot_table = results.pivot_table(
    values='mean_test_score',
    index='param_max_depth',
    columns='param_min_samples_split'
)

# Rysowanie heatmapy
im = plt.imshow(pivot_table, cmap='RdYlGn', aspect='auto')
plt.colorbar(im, label='Accuracy')

# Ustawienia osi
plt.xticks(range(len(pivot_table.columns)), pivot_table.columns)
plt.yticks(range(len(pivot_table.index)), pivot_table.index)
plt.xlabel('min_samples_split', fontsize=12)
plt.ylabel('max_depth', fontsize=12)
plt.title('GridSearchCV - Heatmapa wyników\n(zielone = lepsze, czerwone = gorsze)', fontsize=14)

# Dodanie wartości w komórkach
for i in range(len(pivot_table.index)):
    for j in range(len(pivot_table.columns)):
        text = plt.text(j, i, f'{pivot_table.iloc[i, j]:.3f}',
                       ha="center", va="center", color="black", fontsize=9)

plt.tight_layout()
plt.show()

# ==============================================================================
# WIZUALIZACJA 2: WYKRES SŁUPKOWY TOP 10 KOMBINACJI
# ==============================================================================
plt.figure(figsize=(12, 6))

# Sortowanie wyników
results_sorted = results.sort_values('mean_test_score', ascending=False).head(10)

# Tworzenie etykiet
labels = [f"depth={row['param_max_depth']}\nsplit={row['param_min_samples_split']}"
          for _, row in results_sorted.iterrows()]

# Wykres słupkowy
bars = plt.bar(range(len(results_sorted)), results_sorted['mean_test_score'],
               color='skyblue', edgecolor='navy')

# Podświetlenie najlepszego wyniku
bars[0].set_color('gold')
bars[0].set_edgecolor('orange')
bars[0].set_linewidth(3)

plt.xticks(range(len(results_sorted)), labels, rotation=45, ha='right')
plt.ylabel('Accuracy (CV)', fontsize=12)
plt.title('GridSearchCV - Top 10 kombinacji parametrów', fontsize=14)
plt.ylim([results_sorted['mean_test_score'].min() - 0.01,
          results_sorted['mean_test_score'].max() + 0.01])
plt.axhline(y=results_sorted['mean_test_score'].iloc[0],
            color='red', linestyle='--', alpha=0.5, label='Najlepszy wynik')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# ==============================================================================
# WIZUALIZACJA 3: WPŁYW POSZCZEGÓLNYCH PARAMETRÓW
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Wykres 1: Wpływ max_depth
depth_scores = results.groupby('param_max_depth')['mean_test_score'].mean()
axes[0].plot(depth_scores.index, depth_scores.values, 'o-', linewidth=2, markersize=8, color='blue')
axes[0].set_xlabel('max_depth', fontsize=12)
axes[0].set_ylabel('Średnia accuracy', fontsize=12)
axes[0].set_title('Wpływ max_depth na wyniki', fontsize=13)
axes[0].grid(True, alpha=0.3)
axes[0].fill_between(depth_scores.index, depth_scores.values, alpha=0.3)

# Wykres 2: Wpływ min_samples_split
split_scores = results.groupby('param_min_samples_split')['mean_test_score'].mean()
axes[1].plot(split_scores.index, split_scores.values, 'o-', linewidth=2, markersize=8, color='green')
axes[1].set_xlabel('min_samples_split', fontsize=12)
axes[1].set_ylabel('Średnia accuracy', fontsize=12)
axes[1].set_title('Wpływ min_samples_split na wyniki', fontsize=13)
axes[1].grid(True, alpha=0.3)
axes[1].fill_between(split_scores.index, split_scores.values, alpha=0.3, color='green')

plt.tight_layout()
plt.show()

# ==============================================================================
# PODSUMOWANIE
# ==============================================================================
print("\n" + "="*60)
print("NAJLEPSZE PARAMETRY:", grid_search.best_params_)
print(f"NAJLEPSZY WYNIK (CV): {grid_search.best_score_:.4f}")
print(f"WYNIK NA ZBIORZE TESTOWYM: {grid_search.score(X_test, y_test):.4f}")
print(f"PRZETESTOWANO {len(results)} kombinacji")
print("="*60)
