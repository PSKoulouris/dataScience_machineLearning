from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
import seaborn as sns

digits = datasets.load_digits()
#define the data and the label:
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,       # your features and labels
    test_size=0.2,  # 20% of the data will be used for testing
    random_state=42, # ensures reproducibility
    stratify=y      # optional: preserves the proportion of each class
)

#classifier: Decision Tree
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Detailed report
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=digits.target_names,
            yticklabels=digits.target_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap")
plt.show()



print(X)
print(y)
print(X.shape)
print(y.shape)
print(X_train.shape)
print(X_test.shape)





"""
plt.imshow(digits.images[1], cmap='gray')
plt.title(f"Digit: {digits.target[0]}")
plt.show()

# Number of images to display
num_images = 16

plt.figure(figsize=(8, 8))  # big figure for multiple images

for i in range(num_images):
    plt.subplot(4, 4, i + 1)  # 4x4 grid
    plt.imshow(digits.images[i], cmap='gray')  # show the image
    plt.title(f"{digits.target[i]}")  # show the label
    plt.axis('off')  # hide axes for clarity

plt.tight_layout()
plt.show()
"""

