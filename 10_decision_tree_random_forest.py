# Decision Tree and Random Forest Classification
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
)

tree = DecisionTreeClassifier(max_depth=3, random_state=42)
forest = RandomForestClassifier(n_estimators=100, random_state=42)

for name, model in [("Decision Tree", tree), ("Random Forest", forest)]:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions, target_names=data.target_names))

plt.figure(figsize=(12, 7))
plot_tree(tree.fit(X_train, y_train), feature_names=data.feature_names,
          class_names=data.target_names, filled=True)
plt.title("Decision Tree")
plt.show()

plt.bar(data.feature_names, forest.fit(X_train, y_train).feature_importances_)
plt.xticks(rotation=45, ha="right")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()
