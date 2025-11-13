import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

my_log = LogisticRegression()






data = pd.read_csv('banking.csv', sep = ',')
data = data.dropna()

head = data.head()
listColumns = list(data.columns)
dataInformation = data.info()

numerical_inputs = data.drop['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']


my_log.fit(numerical_inputs, data['y'])
my_log.coef_


#print(head)
#print(listColumns)
#print(dataInformation)

print(my_log.coef_)