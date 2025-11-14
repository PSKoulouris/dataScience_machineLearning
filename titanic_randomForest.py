"""
TITANIC SURVIVAL PREDICTION WITH RANDOM FOREST

"""

import pandas as pd
from io import StringIO
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Read file lines
with open("titanic.csv", "r", encoding="utf-8") as f:
    lines = [line.strip().strip('"').replace('""', '"') for line in f]

# Clean CSV and read into pandas
cleaned_csv = StringIO("\n".join(lines))
df = pd.read_csv(cleaned_csv, sep=",")


cols = ['Survived','Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
df_select = df[cols]

# Drop Cabin (too many missing values)
df_select = df_select.drop(columns=['Cabin'])


# Fill missing Age with median
df_select['Age'] = df_select['Age'].fillna(df_select['Age'].median())

# Fill missing Embarked with mode
df_select['Embarked'] = df_select['Embarked'].fillna(df_select['Embarked'].mode()[0])

# Sex → 0 = male, 1 = female
df_select['Sex'] = df_select['Sex'].map({'male': 0, 'female': 1})

# Embarked → one-hot encoding (drop first to avoid dummy variable trap)
df_select = pd.get_dummies(df_select, columns=['Embarked'], drop_first=True)

X = df_select.drop(columns=['Survived'])
y = df_select['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create Random Forest with 100 trees
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

#Predictions: 
y_pred = rf.predict(X_test)
#Performance: 
# Accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#visualize confusion matrix:
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Died', 'Predicted Survived'],
            yticklabels=['Actual Died', 'Actual Survived'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix")
plt.show()

#Feature importance:

importances = rf.feature_importances_
feature_names = X.columns

# Combine and sort
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print(feat_imp)

# Optional: plot
feat_imp.plot(kind='bar', figsize=(10,5), title="Feature Importance")
plt.show()

