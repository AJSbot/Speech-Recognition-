import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits

print("STEP 1: Loading Dataset")

# Load dataset
data = load_digits()

X = data.data
y = data.target

print("Dataset Loaded Successfully")
print("Total Samples:", X.shape[0])
print("Number of Features:", X.shape[1])

print("\nSTEP 2: Splitting Dataset into Training and Testing")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

print("\nSTEP 3: Training the Model")

# Train model
model = GaussianNB()
model.fit(X_train, y_train)

print("Model Training Completed")

print("\nSTEP 4: Making Predictions")

# Predictions
y_pred = model.predict(X_test)

print("Prediction Completed")

print("\nSTEP 5: Calculating Accuracy")

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy * 100, 2), "%")

print("\nSTEP 6: Phoneme (Class) Prediction Probabilities")

# Prediction probabilities
probabilities = model.predict_proba(X_test)

# Display probabilities for first 5 samples
for i in range(5):
    print("\nSample", i+1)
    print("Actual Label:", y_test[i])
    print("Predicted Label:", y_pred[i])
    print("Prediction Probabilities:")

    for label, prob in enumerate(probabilities[i]):
        print("Class", label, ":", round(prob, 3))

print("\nSTEP 7: Generating Confusion Matrix")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

print("\nSTEP 8: Displaying Confusion Matrix")

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Confusion Matrix")
plt.show()