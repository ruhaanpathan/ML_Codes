# Support Vector Machine Classification
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
)

model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0))
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions, target_names=data.target_names))
ConfusionMatrixDisplay.from_predictions(
    y_test, predictions, display_labels=data.target_names
)
plt.title("SVM Confusion Matrix")
plt.show()
