import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data      # Features: sepal length, sepal width, petal length, petal width
y = iris.target    # Original target: 0 = Setosa, 1 = Versicolor, 2 = Virginica

# 2. Convert to a binary classification problem: Setosa vs. not Setosa
#    We'll assign 1 if the flower is Setosa (original label 0), and 0 otherwise.
y_bin = (y == 0).astype(int)

# Optional: Create a DataFrame to inspect the data
df = pd.DataFrame(X, columns=iris.feature_names)
df['is_setosa'] = y_bin
#print(df.head())

# 3. Split the data into training (70%) and test (30%) sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y_bin, test_size=0.3, random_state=42, stratify=y_bin
)

# 4. Scale the features for numerical stability using StandardScaler.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Initialize and train a Logistic Regression model.
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train_scaled, y_train)

# 6. Make predictions on the test set.
y_pred = logreg.predict(X_test_scaled)

# 7. Evaluate the model using several metrics.
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Not Setosa", "Setosa"])

print("Model Evaluation Metrics:")
print("-------------------------")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}\n")
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)

# Optional: Plot decision boundary (for visualization using first 2 features)
# Note: This is only for illustration since our model was trained on all 4 features.
def plot_decision_boundary(X, y, model, title="Decision Boundary (first 2 features)"):
    # Create a mesh grid based on the first two features.
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # For plotting, we train a new model on only the first two features.
    model_2d = LogisticRegression(random_state=42)
    model_2d.fit(X[:, :2], y)
    
    # Predict over the grid.
    Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot contour and training points.
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.Paired)
    plt.xlabel("Feature 1 (scaled)")
    plt.ylabel("Feature 2 (scaled)")
    plt.title(title)
    plt.show()

# Plot decision boundary for the first two features (using training data).
plot_decision_boundary(X_train_scaled, y_train, logreg)
