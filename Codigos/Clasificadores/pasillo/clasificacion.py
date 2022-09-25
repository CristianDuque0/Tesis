#Bibliotecas
import array
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import pandas as pd
import csv
import os
import argparse
from random import randrange
from sklearn import metrics
from sklearn . neighbors import KNeighborsClassifier
from sklearn . linear_model import LogisticRegression 
from sklearn . model_selection import train_test_split
from sklearn . tree import DecisionTreeClassifier
from sklearn . discriminant_analysis import LinearDiscriminantAnalysis
from sklearn . naive_bayes import GaussianNB
from sklearn . svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn . model_selection import KFold
from sklearn . model_selection import cross_val_score
from sklearn . metrics import accuracy_score
from sklearn . metrics import confusion_matrix
from sklearn . metrics import classification_report

#Leer csv con el vector de caracteristicas
df=pd.read_csv('vgraficar.csv')

#Informacion del csv
print(df.head(60))
print(df.describe())
print(df.groupby('genero').size())
print(df.shape)

#Grafica de cajas y bigotes media
plt.figure(figsize=(15 ,20))
plt.subplot(9,2,1)
sns.boxplot(x='genero',y='mCt',data=df)
plt.subplot(9,2,2)
sns.boxplot(x='genero',y='mAe',data=df)
plt.subplot(9,2,3)
sns.boxplot(x='genero',y='mFt',data=df)
plt.subplot(9,2,4)
sns.boxplot(x='genero',y='mZt',data=df)
plt.subplot(9,2,5)
sns.boxplot(x='genero',y='mMFCC1',data=df)
plt.subplot(9,2,6)
sns.boxplot(x='genero',y='mMFCC2',data=df)
plt.subplot(9,2,7)
sns.boxplot(x='genero',y='mMFCC3',data=df)
plt.subplot(9,2,8)
sns.boxplot(x='genero',y='mMFCC4',data=df)
plt.subplot(9,2,9)
sns.boxplot(x='genero',y='mMFCC5',data=df)
plt.savefig('cybmean.pdf',dpi=400,bbox_inches='tight',pad_inches=0.1)

#Grafica de cajas y bigotes desviacion estandar
plt.subplot(9,2,1)
sns.boxplot(x='genero',y='stdCt',data=df)
plt.subplot(9,2,2)
sns.boxplot(x='genero',y='stdAe',data=df)
plt.subplot(9,2,3)
sns.boxplot(x='genero',y='stdFt',data=df)
plt.subplot(9,2,4)
sns.boxplot(x='genero',y='stdZt',data=df)
plt.subplot(9,2,5)
sns.boxplot(x='genero',y='stdMFCC1',data=df)
plt.subplot(9,2,6)
sns.boxplot(x='genero',y='stdMFCC2',data=df)
plt.subplot(9,2,7)
sns.boxplot(x='genero',y='stdMFCC3',data=df)
plt.subplot(9,2,8)
sns.boxplot(x='genero',y='stdMFCC4',data=df)
plt.subplot(9,2,9)
sns.boxplot(x='genero',y='stdMFCC5',data=df)
plt.savefig('cybstd.pdf',dpi=400,bbox_inches='tight',pad_inches=0.1)

#Diagrama de dispersion
sns.pairplot(df,hue='genero',height=2,diag_kind="kde")
sns.pairplot(df,hue='genero')
plt.savefig('dispersion.pdf',dpi =400,bbox_inches='tight',pad_inches=0.1)

#Determinar valores de prueba y de entrenamiento
X=df[['mCt','mAe','mFt','mZt','mMFCC1','mMFCC2','mMFCC3','mMFCC4','mMFCC5','stdCt','stdAe','stdFt','stdZt','stdMFCC1','stdMFCC2','stdMFCC3','stdMFCC4','stdMFCC5']].values
Y=df['genero'].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Entrenamiento 
results=[]
names=[]
models=[]
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC()))
models.append(('KNC',KNeighborsClassifier(2)))
for name, model in models:
    kf=KFold(n_splits=10)
    clf=model
    clf.fit(X_train,y_train)
    score=clf.score(X_train,y_train)
    print("--------- Modelo ",model,"----------")
    print("Metrica del modelo",score)
    results.append(score)
    names.append(name)
    scores=cross_val_score(clf,X_train,y_train,cv=kf,scoring ="accuracy")
    print("Metricas cross_validation",scores)
    print("Media de cross_validation",scores.mean())
    preds=clf.predict(X_test)
    score_pred=metrics.accuracy_score(y_test,preds)
    print("Metrica en Test",score_pred)

#Imprimir matriz de confusion LDA
lda=LinearDiscriminantAnalysis()
lda.fit(X_train,y_train)
predictions=lda.predict(X_test)
print(accuracy_score(y_test,predictions))
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
matlda=confusion_matrix(y_test,predictions)
fig,ax=plt.subplots(figsize=(10,5))
ax.matshow(matlda)
plt.title('Matriz de confusion LDA', fontsize=20)
for(i,j), z in np.ndenumerate(matlda):
    ax.text(j,i,'{:0.0f}'.format(z), ha='center',va='center')
plt.savefig('matrizlda.png',dpi =400,bbox_inches='tight',pad_inches=0.1)

#Imprimir matriz de confusion CART
cart=DecisionTreeClassifier()
cart.fit(X_train,y_train)
predictions=cart.predict(X_test)
print(accuracy_score(y_test,predictions))
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
matcart=confusion_matrix(y_test,predictions)
fig,ax=plt.subplots(figsize=(10,5))
ax.matshow(matcart)
plt.title('Matriz de confusion CART', fontsize=20)
for(i,j), z in np.ndenumerate(matlda):
    ax.text(j,i,'{:0.0f}'.format(z), ha='center',va='center')
plt.savefig('matrizcart.png',dpi =400,bbox_inches='tight',pad_inches=0.1)

#Imprimir matriz de confusion NB
nb=GaussianNB()
nb.fit(X_train,y_train)
predictions=nb.predict(X_test)
print(accuracy_score(y_test,predictions))
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
matnb=confusion_matrix(y_test,predictions)
fig,ax=plt.subplots(figsize=(10,5))
ax.matshow(matnb)
plt.title('Matriz de confusion NB', fontsize=20)
for(i,j), z in np.ndenumerate(matnb):
    ax.text(j,i,'{:0.0f}'.format(z), ha='center',va='center')
plt.savefig('matriznb.png',dpi =400,bbox_inches='tight',pad_inches=0.1)

#Imprimir matriz de confusion SVC
svc=SVC()
svc.fit(X_train,y_train)
predictions=svc.predict(X_test)
print(accuracy_score(y_test,predictions))
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
matsvc=confusion_matrix(y_test,predictions)
fig,ax=plt.subplots(figsize=(10,5))
ax.matshow(matsvc)
plt.title('Matriz de confusion SVC', fontsize=20)
for(i,j), z in np.ndenumerate(matsvc):
    ax.text(j,i,'{:0.0f}'.format(z), ha='center',va='center')
plt.savefig('matrizsvc.png',dpi =400,bbox_inches='tight',pad_inches=0.1)

#Imprimir matriz de confusion KNN
knn=KNeighborsClassifier(2)
knn.fit(X_train,y_train)
predictions=knn.predict(X_test)
print(accuracy_score(y_test,predictions))
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
matknn=confusion_matrix(y_test,predictions)
fig,ax=plt.subplots(figsize=(10,5))
ax.matshow(matknn)
plt.title('Matriz de confusion KNN', fontsize=20)
for(i,j), z in np.ndenumerate(matknn):
    ax.text(j,i,'{:0.0f}'.format(z), ha='center',va='center')
plt.savefig('matrizknn.png',dpi =400,bbox_inches='tight',pad_inches=0.1)