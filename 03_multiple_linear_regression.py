# Multiple Linear Regression using the Diabetes dataset
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = load_diabetes()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
print("MSE:", mean_squared_error(y_test, predictions))
print("R2 score:", r2_score(y_test, predictions))

plt.scatter(y_test, predictions)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("Multiple Linear Regression")
plt.show()
