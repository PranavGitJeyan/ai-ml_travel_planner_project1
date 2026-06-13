import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

print("==================================================")
print("     MACHINE LEARNING MODEL EVALUATION METRICS     ")
print("==================================================\n")

# 1. Recreate the training dataset
# Features: [Price, Duration]
X_train = np.array([
    [20, 0], [35, 4], [250, 0], [180, 1.5], 
    [15, 10], [500, 0], [45, 0], [130, 1]
])
# Labels: 1 = Approved Budget, 0 = Rejected Luxury
y_train = np.array([1, 1, 0, 0, 1, 0, 1, 0])

# 2. Train the Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# 3. Generate Predictions on the data to evaluate accuracy
y_pred = model.predict(X_train)

# 4. Print the Classification Report (Accuracy, Precision, Recall, F1)
print("📊 CLASSIFICATION REPORT:")
print(classification_report(y_train, y_pred, target_names=["Rejected (0)", "Approved (1)"]))
print("-" * 50)

# 5. Print the Raw Confusion Matrix
print("\n🧮 CONFUSION MATRIX VALUES:")
cm = confusion_matrix(y_train, y_pred)
print(cm)
print(f"True Negatives (Luxury correctly rejected): {cm[0][0]}")
print(f"False Positives: {cm[0][1]}")
print(f"False Negatives: {cm[1][0]}")
print(f"True Positives (Budget correctly approved): {cm[1][1]}")
print("-" * 50)

# 6. Generate and Save a Visual Confusion Matrix Plot
fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Rejected", "Approved"])
disp.plot(cmap="Blues", ax=ax, values_format="d")
plt.title("Model Confusion Matrix")

# Save the chart as an image in your folder
plt.savefig("confusion_matrix.png", bbox_inches="tight")
plt.close()

print("\n🎨 SUCCESS: Confusion matrix plot saved locally as 'confusion_matrix.png'!")
print("==================================================")