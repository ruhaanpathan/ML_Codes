# K-Nearest Neighbors Classification
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
)

model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions, target_names=data.target_names))
ConfusionMatrixDisplay.from_predictions(
    y_test, predictions, display_labels=data.target_names
)
plt.title("KNN Confusion Matrix")
plt.show()

sample = [[5.1, 3.5, 1.4, 0.2]]
print("Predicted class:", data.target_names[model.predict(sample)[0]])
