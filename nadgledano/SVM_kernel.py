import pandas as pd
from sklearn import metrics,preprocessing,model_selection
from sklearn import svm
import matplotlib.pyplot as plt

df=pd.read_csv('card_transdata.csv')
X=df.drop(['fraud'],axis=1).iloc[:80000]
y=df['fraud'].iloc[:80000]

X_train_valid,X_test,y_train_valid,y_test=model_selection.train_test_split(X,y,train_size=0.7,stratify=y)
X_train,X_valid,y_train,y_valid=model_selection.train_test_split(X_train_valid,y_train_valid,train_size=0.7,stratify=y_train_valid)


scaler=preprocessing.StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_valid=scaler.transform(X_valid)
X_test=scaler.transform(X_test)

Cs = [0.01,0.1, 1,10]
gammas = [0.01,0.1, 1,10]

F1_skor_najbolji=0
best_c=None
best_gamma=None

# treniranje na trening skupu i ocenjuje se na validacionom skupu
for i in gammas:
    for j in Cs:
        model=svm.SVC(kernel='rbf',gamma=i,C=j)
        print(i, j, "\n")
        model.fit(X_train,y_train)
        y_pred_valid=model.predict(X_valid)
        F1_trenutni=metrics.f1_score(y_valid,y_pred_valid)
        if F1_skor_najbolji<F1_trenutni:
            best_gamma=i
            best_c=j
            F1_skor_najbolji=F1_trenutni


print("Najbolji parametri:",best_c,best_gamma)

best_model=svm.SVC(kernel='rbf',gamma=best_gamma,C=best_c)
best_model.fit(X_train,y_train)

y_pred=best_model.predict(X_test)
F1_score=metrics.f1_score(y_test,y_pred)

print(metrics.classification_report(y_test,y_pred))
print(F1_score)
print(metrics.precision_score(y_test,y_pred))
print(metrics.recall_score(y_test,y_pred))