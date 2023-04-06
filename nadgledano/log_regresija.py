import numpy as np
import pandas as pd
from sklearn import metrics, preprocessing, model_selection
from sklearn import linear_model

df = pd.read_csv('card_transdata.csv')

X = df.drop(['fraud'], axis=1)
y = df['fraud']

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, stratify=y)

scaler = preprocessing.StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = linear_model.LogisticRegression(solver='lbfgs', C=0.1, max_iter=500, penalty='l2')
model.fit(X_train, y_train)

y_test_pred = model.predict(X_test)

print(metrics.precision_score(y_test, y_test_pred))
print(metrics.recall_score(y_test, y_test_pred))
print(metrics.f1_score(y_test, y_test_pred))
