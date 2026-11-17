"""Task 2: Compare linear and quadratic models for yacht resistance."""
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

yachts = pd.read_csv("yacht_hydrodynamics.data", sep=r"\s+", header=None)
X = yachts[[5]]  # Sixth column: Froude number
y = yachts[6]     # Seventh column: residuary resistance

linear_model = LinearRegression()
quadratic_model = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("linear", LinearRegression()),
])
folds = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = "neg_root_mean_squared_error"

linear_rmse = -cross_val_score(linear_model, X, y, cv=folds, scoring=scoring)
quadratic_rmse = -cross_val_score(quadratic_model, X, y, cv=folds, scoring=scoring)

print("Linear fold RMSEs:", linear_rmse.round(2))
print(f"Linear mean RMSE: {linear_rmse.mean():.2f}")
print("Quadratic fold RMSEs:", quadratic_rmse.round(2))
print(f"Quadratic mean RMSE: {quadratic_rmse.mean():.2f}")
print("Better model:", "Quadratic" if quadratic_rmse.mean() < linear_rmse.mean() else "Linear")
