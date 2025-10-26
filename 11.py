from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. WCZYTANIE DANYCH
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. DEFINICJA PARAMETRÓW (szeroki zakres!)
param_distributions = {
    'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'min_samples_split': [2, 3, 5, 7, 10, 15, 20],
    'min_samples_leaf': [1, 2, 3, 4, 5]
}
# Możliwych kombinacji: 9 × 7 × 5 = 315

# 3. RANDOMIZEDSEARCHCV
model = DecisionTreeClassifier(random_state=42)
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=30,  # testujemy tylko 30 losowych kombinacji!
    cv=5,
    scoring='accuracy',
    random_state=42,
    return_train_score=True
)

print("Losowe przeszukiwanie parametrów...")
random_search.fit(X_train, y_train)

# 4. PRZYGOTOWANIE DANYCH
results = pd.DataFrame(random_search.cv_results_)

# ==============================================================================
# WIZUALIZACJA 1: WYNIKI POSORTOWANE (WODOSPAD)
# ==============================================================================
plt.figure(figsize=(14, 6))

# Sortowanie
results_sorted = results.sort_values('mean_test_score', ascending=False)
x_pos = range(len(results_sorted))

# Wykres
colors = ['gold' if i == 0 else 'lightcoral' if i >= len(results_sorted)-5
          else 'skyblue' for i in range(len(results_sorted))]

bars = plt.bar(x_pos, results_sorted['mean_test_score'], color=colors,
               edgecolor='navy', linewidth=1)

# Oznaczenie najlepszego i najgorszych
bars[0].set_edgecolor('orange')
bars[0].set_linewidth(3)

plt.xlabel('Kombinacja parametrów (posortowane)', fontsize=12)
plt.ylabel('Accuracy (CV)', fontsize=12)
plt.title(f'RandomizedSearchCV - Wszystkie {len(results)} przetestowane kombinacje\n(złoty=najlepszy, czerwone=najgorsze)', fontsize=14)
plt.axhline(y=results_sorted['mean_test_score'].iloc[0], color='red',
            linestyle='--', alpha=0.5, label=f'Najlepszy: {results_sorted["mean_test_score"].iloc[0]:.4f}')
plt.axhline(y=results_sorted['mean_test_score'].mean(), color='blue',
            linestyle='--', alpha=0.5, label=f'Średnia: {results_sorted["mean_test_score"].mean():.4f}')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# ==============================================================================
# WIZUALIZACJA 2: TOP 10 KOMBINACJI Z WARTOŚCIAMI PARAMETRÓW
# ==============================================================================
plt.figure(figsize=(14, 7))

top10 = results_sorted.head(10)

# Tworzenie szczegółowych etykiet
labels = []
for idx, row in top10.iterrows():
    label = f"depth={row['param_max_depth']}\n"
    label += f"split={row['param_min_samples_split']}\n"
    label += f"leaf={row['param_min_samples_leaf']}"
    labels.append(label)

# Wykres
bars = plt.bar(range(len(top10)), top10['mean_test_score'],
               color=plt.cm.RdYlGn(np.linspace(0.5, 0.9, len(top10))),
               edgecolor='black', linewidth=1.5)

# Dodanie wartości na słupkach
for i, (bar, score) in enumerate(zip(bars, top10['mean_test_score'])):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{score:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.xticks(range(len(top10)), labels, rotation=0, ha='center', fontsize=9)
plt.ylabel('Accuracy (CV)', fontsize=12)
plt.title('RandomizedSearchCV - Top 10 najlepszych kombinacji', fontsize=14)
plt.ylim([top10['mean_test_score'].min() - 0.02,
          top10['mean_test_score'].max() + 0.015])
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# ==============================================================================
# WIZUALIZACJA 3: ROZKŁAD WYNIKÓW (HISTOGRAM + BOX PLOT)
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(results['mean_test_score'], bins=15, color='skyblue',
             edgecolor='navy', alpha=0.7)
axes[0].axvline(results['mean_test_score'].mean(), color='red',
                linestyle='--', linewidth=2, label=f'Średnia: {results["mean_test_score"].mean():.4f}')
axes[0].axvline(random_search.best_score_, color='gold',
                linestyle='--', linewidth=2, label=f'Najlepszy: {random_search.best_score_:.4f}')
axes[0].set_xlabel('Accuracy', fontsize=12)
axes[0].set_ylabel('Liczba kombinacji', fontsize=12)
axes[0].set_title('Rozkład wyników (histogram)', fontsize=13)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Box plot
box = axes[1].boxplot([results['mean_test_score']],
                       vert=True, patch_artist=True, widths=0.5)
box['boxes'][0].set_facecolor('lightblue')
box['boxes'][0].set_edgecolor('navy')
box['boxes'][0].set_linewidth(2)
box['medians'][0].set_color('red')
box['medians'][0].set_linewidth(2)

axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title('Box plot wyników', fontsize=13)
axes[1].set_xticklabels(['Wszystkie\nkombinacje'])
axes[1].grid(axis='y', alpha=0.3)

# Dodanie punktu dla najlepszego wyniku
axes[1].scatter([1], [random_search.best_score_], color='gold',
                s=200, zorder=5, edgecolor='orange', linewidth=2,
                label='Najlepszy', marker='*')
axes[1].legend()

plt.tight_layout()
plt.show()

# ==============================================================================
# WIZUALIZACJA 4: PORÓWNANIE Z PRZESTRZENIĄ PARAMETRÓW
# ==============================================================================
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Wyciągnięcie wartości parametrów
depths = results['param_max_depth'].values
splits = results['param_min_samples_split'].values
scores = results['mean_test_score'].values

# Scatter plot 3D
scatter = ax.scatter(depths, splits, scores,
                     c=scores, cmap='RdYlGn',
                     s=100, alpha=0.6, edgecolors='black')

# Oznaczenie najlepszego punktu
best_idx = results['mean_test_score'].idxmax()
ax.scatter([results.loc[best_idx, 'param_max_depth']],
          [results.loc[best_idx, 'param_min_samples_split']],
          [results.loc[best_idx, 'mean_test_score']],
          color='gold', s=300, marker='*', edgecolors='orange',
          linewidths=2, label='Najlepszy')

ax.set_xlabel('max_depth', fontsize=11)
ax.set_ylabel('min_samples_split', fontsize=11)
ax.set_zlabel('Accuracy', fontsize=11)
ax.set_title('RandomizedSearchCV - Przestrzeń parametrów 3D\n(30 losowych punktów z 315 możliwych)', fontsize=13)
plt.colorbar(scatter, label='Accuracy', pad=0.1)
ax.legend()

plt.tight_layout()
plt.show()

# ==============================================================================
# PODSUMOWANIE
# ==============================================================================
print("\n" + "="*60)
print("NAJLEPSZE PARAMETRY:", random_search.best_params_)
print(f"NAJLEPSZY WYNIK (CV): {random_search.best_score_:.4f}")
print(f"WYNIK NA ZBIORZE TESTOWYM: {random_search.score(X_test, y_test):.4f}")
print(f"PRZETESTOWANO: {len(results)} z 315 możliwych kombinacji ({len(results)/315*100:.1f}%)")
print(f"ŚREDNI WYNIK: {results['mean_test_score'].mean():.4f}")
print(f"ODCHYLENIE: {results['mean_test_score'].std():.4f}")
print("="*60)
