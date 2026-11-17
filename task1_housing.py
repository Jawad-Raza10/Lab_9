"""Task 1: Estimate housing price per unit area from MRT distance."""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

homes = pd.read_csv("real_estate.csv")
homes["distance_100m"] = homes["X3 distance to the nearest MRT station"] / 100
X = homes[["distance_100m"]]
y = homes["Y house price of unit area"]

model = LinearRegression()
model.fit(X, y)
print(f"Slope: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
at_500m = model.predict(pd.DataFrame({"distance_100m": [5]}))[0]
print(f"Predicted price at 500 m: {at_500m:.4f} price units")

line_x = pd.DataFrame({"distance_100m": [X.min().iloc[0], X.max().iloc[0]]})
line_y = model.predict(line_x)
plt.figure(figsize=(8, 5))
plt.scatter(X["distance_100m"], y, alpha=0.55, label="Observed transactions")
plt.plot(line_x["distance_100m"], line_y, color="red", linewidth=2, label="Fitted line")
plt.xlabel("Distance to nearest MRT station (100 m)")
plt.ylabel("House price of unit area (10,000 TWD per ping)")
plt.title("Housing Price vs. Distance to MRT Station")
plt.legend()
plt.tight_layout()
plt.savefig("task1_housing.png", dpi=300)
plt.show()
