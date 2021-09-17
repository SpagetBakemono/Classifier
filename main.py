# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 15:08:27 2021

@author: Aditya
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

url = "https://raw.githubusercontent.com/JWarmenhoven/ISLR-python/master/Notebooks/Data/Smarket.csv"

df = pd.read_csv(url)

print(df.head())

train_targets = df[df.Year < 2005].Direction
test_targets = df[df.Year == 2005].Direction
train_labels = df[df.Year < 2005][["Lag1", "Lag2", "Lag3", "Lag4", "Lag5", "Volume"]]
test_labels = df[df.Year == 2005][["Lag1", "Lag2", "Lag3", "Lag4", "Lag5", "Volume"]]

lags = df[["Lag1", "Lag2", "Lag3", "Lag4", "Lag5", "Today"]]

figure, axis = plt.subplots(1, 2)

covMatrix = lags.corr()
sns.heatmap(covMatrix, annot=True, cmap="YlGnBu", ax=axis[0])
axis[1].plot(df["Volume"])
plt.show()

lr = LogisticRegression()
lr.fit(train_labels, train_targets)
y_preds = lr.predict(test_labels)
lr_cnf_matrix = metrics.confusion_matrix(test_targets, y_preds)

y_train_preds = lr.predict(train_labels)
lr_cnf_matrix2 = metrics.confusion_matrix(train_targets, y_train_preds)

fig, ax = plt.subplots()
sns.heatmap(pd.DataFrame(lr_cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

print("Accuracy: ", metrics.accuracy_score(test_targets, y_preds))





