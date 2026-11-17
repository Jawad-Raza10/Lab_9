"""Task 3: Tune a polynomial Ridge model for concrete strength."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

concrete = pd.read_csv("concrete_strength.csv").drop_duplicates()
target = "Concrete compressive strength"
X = concrete.drop(columns=target)
y = concrete[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
folds = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = "neg_root_mean_squared_error"

baseline_scores = -cross_val_score(
    LinearRegression(), X_train, y_train, cv=folds, scoring=scoring
)
print("Linear baseline fold RMSEs:", baseline_scores.round(2))
print(f"Linear baseline mean CV RMSE: {baseline_scores.mean():.2f} MPa")

ridge_model = Pipeline([
    ("poly", PolynomialFeatures(include_bias=False)),
    ("scale", StandardScaler()),
    ("ridge", Ridge()),
])
grid = {
    "poly__degree": [1, 2, 3],
    "ridge__alpha": [0.01, 0.1, 1, 10, 100],
}
search = GridSearchCV(
    ridge_model, grid, cv=folds, scoring=scoring, refit=True
)
search.fit(X_train, y_train)

results = search.cv_results_
top_indices = np.argsort(results["rank_test_score"])[:3]
print("\nTop three grid-search settings:")
for i in top_indices:
    print(
        f"Rank {results['rank_test_score'][i]}: "
        f"{results['params'][i]}, "
        f"mean CV RMSE={-results['mean_test_score'][i]:.2f} MPa"
    )

predictions = search.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, predictions))
test_mae = mean_absolute_error(y_test, predictions)
print("Best parameters:", search.best_params_)
print(f"Best mean CV RMSE: {-search.best_score_:.2f} MPa")
print(f"Test RMSE: {test_rmse:.2f} MPa")
print(f"Test MAE: {test_mae:.2f} MPa")

plt.figure(figsize=(7, 6))
plt.scatter(y_test, predictions, alpha=0.65)
limits = [
    min(y_test.min(), predictions.min()),
    max(y_test.max(), predictions.max()),
]
plt.plot(limits, limits, "r--", label="Ideal: y = x")
plt.xlim(limits)
plt.ylim(limits)
plt.xlabel("Actual concrete strength (MPa)")
plt.ylabel("Predicted concrete strength (MPa)")
plt.title("Actual vs. Predicted Concrete Strength")
plt.legend()
plt.tight_layout()
plt.savefig("task3_concrete.png", dpi=300)
plt.show()
