import numpy as np
import pandas as pd
from io import StringIO
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.neighbors import NearestNeighbors


# Step 1: read the file lines
with open("titanic.csv", "r", encoding="utf-8") as f:
    lines = [line.strip().strip('"').replace('""', '"') for line in f]

# Step 2: join lines into a string buffer
cleaned_csv = StringIO("\n".join(lines))

# Step 3: read with pandas
df = pd.read_csv(cleaned_csv, sep=",")

# select columns
cols = ['Survived','Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
df_select = df[cols]
df_select = df_select.drop(columns=['Cabin'])

# fix missing values
df_select['Age'] = df_select['Age'].fillna(df_select['Age'].median())
df_select['Embarked'] = df_select['Embarked'].fillna(df_select['Embarked'].mode()[0])

# encode categorical variables
df_select['Sex'] = df_select['Sex'].map({'male': 0, 'female': 1})
df_select = pd.get_dummies(df_select, columns=['Embarked'], drop_first=True)

# define X, y
X = df_select.drop(columns=['Survived'])
y = df_select['Survived']

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# predict
y_pred = knn.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
plt.show()

# Fit nearest-neighbors model (same scaling as KNN)
nn = NearestNeighbors(n_neighbors=5)
nn.fit(X_train_scaled)

# Compute distances from test points to their 5 nearest neighbors
distances, indices = nn.kneighbors(X_test_scaled)

# Sort distances to visualize outliers better
sorted_distances = np.sort(distances[:, 4])  # distance to the 5th neighbor

plt.figure(figsize=(7,4))
plt.plot(sorted_distances)
plt.title("K-Nearest Neighbor Distance Plot (k = 5)")
plt.xlabel("Test Samples (sorted)")
plt.ylabel("Distance to 5th Nearest Neighbor")
plt.grid(True)
plt.show()



"""
#separte df into men and women into new dataframes: df_men and df_women
df_men = df_select[df_select['Sex'] == 'male']
df_women = df_select[df_select['Sex'] == 'female']
"""




"""
print(df.head())
print(df_select.head())
print(df.info())
print(df_select.describe())
print(df_select.isnull().sum())
print(df_men.head())
print(df_women.head())
"""


