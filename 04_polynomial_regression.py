# Polynomial Regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

np.random.seed(42)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = 2 + 3 * X[:, 0] - 1.5 * X[:, 0]**2 + np.random.normal(0, 2, 100)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, predictions))
print("R2 score:", r2_score(y_test, predictions))

x_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
plt.scatter(X, y, label="Data")
plt.plot(x_plot, model.predict(x_plot), color="red", label="Polynomial fit")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Polynomial Regression (degree 2)")
plt.legend()
plt.show()
