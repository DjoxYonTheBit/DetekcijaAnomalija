import pandas as pd
from sklearn import metrics,preprocessing,model_selection
from sklearn import svm
import matplotlib.pyplot as plt

df=pd.read_csv('card_transdata.csv')
X=df.drop(['fraud'],axis=1).iloc[:30000]
y=df['fraud'].iloc[:30000]

X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,train_size=0.67,stratify=y)

scaler=preprocessing.StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

model=svm.LinearSVC(loss='hinge')
model.fit(X_train,y_train)

y_pred=model.predict(X_test)
print(metrics.precision_score(y_test,y_pred))
print(metrics.recall_score(y_test,y_pred))