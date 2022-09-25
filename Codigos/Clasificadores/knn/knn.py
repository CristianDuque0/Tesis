import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

dataframe = pd.read_csv("vgraficar.csv")	
X = dataframe[['mCt','mAe','mFt','mZt','mMFCC1','mMFCC2','mMFCC3','mMFCC4','mMFCC5','stdCt','stdAe','stdFt','stdZt','stdMFCC1','stdMFCC2','stdMFCC3','stdMFCC4','stdMFCC5']].values
y = dataframe['genero'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
n_neighbors = 7

knn = KNeighborsClassifier(n_neighbors)
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test, y_test))) 
     
pred = knn.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
matriz=(confusion_matrix(y_test, pred))
plt.figure(figsize=(5,5))
plt.imshow(matriz)
plt.show()
