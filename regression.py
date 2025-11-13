import numpy as np
import pandas as pd
from io import StringIO
import seaborn as sns

# Step 1: read the file lines
with open("titanic.csv", "r", encoding="utf-8") as f:
    lines = [line.strip().strip('"').replace('""', '"') for line in f]

# Step 2: join lines into a string buffer
cleaned_csv = StringIO("\n".join(lines))

# Step 3: read with pandas
df = pd.read_csv(cleaned_csv, sep=",")

description = df.describe()

#select columns of interest for analysis: df_select
colonnes=['Survived','Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
df_select=df[colonnes]

#separte df into men and women into new dataframes: df_men and df_women
df_men = df_select[df_select['Sex'] == 'male']
df_women = df_select[df_select['Sex'] == 'female']




print(df.head())
print(df_select.head())
print(df.info())
print(df_select.describe())
print(df_select.isnull().sum())
print(df_men.head())
print(df_women.head())


