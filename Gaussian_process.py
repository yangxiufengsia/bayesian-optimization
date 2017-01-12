from rdkit import Chem
import csv
import numpy as np
from rdkit.Chem import Descriptors
from numpy import matrix
from numpy.linalg import inv
from math import *
import matplotlib.pyplot as plt
xall=np.linspace(-20,20,num=300)
## Gaussian noise to the function of y
c=np.random.normal(0,1,1000)
#print c.shape
#yall = np.sin(xall)*np.cos(xall)* (1-abs(xall-3)*0.1)
yall = np.sin(xall)* (1-abs(xall-3)*0.1)
#plt.plot(xall,yall)
#plt.show()
#print yall.shape
index=np.random.permutation(range(len(xall)))
n=len(xall)
idtr=200
idtr_l=99
xtrain=xall[index][0:idtr]
xtest=xall[index][idtr+1:n]
ytrain=yall[index][0:idtr]
ytest=yall[index][idtr+1:n]
xtr=np.matrix(xtrain)
xte=np.matrix(xtest)
ytr=(np.matrix(ytrain)).T
yte=(np.matrix(ytest)).T
print xtr.shape
print ytr.shape
### assign prioir to weight
'''w~ N(0,1)'''
### assign Gaussian to noise
''' noise~N(0,1)'''
### posterior calculcation

'''kernel matrix calculation'''

ktr=np.zeros((idtr,idtr))
kte=np.zeros((idtr_l,idtr))
kte_train=np.zeros((idtr,idtr))
kte_test=np.zeros((idtr_l,idtr_l))
kte_kte=np.zeros((idtr,idtr_l))
### training kernel matrix calculation
for i in range(0,idtr):
    for j in range(i,idtr):
        ktr[i,j]=exp(-0.5*(xtr[0,i]-xtr[0,j])**2)
        #print ktr[i,j]
        ktr[j,i]=ktr[i,j]

### testing kernel matrix calculation
for i in range(0,idtr_l):
    for j in range(0,idtr_l):
        kte[i,j]=exp(-0.5*(xte[0,i]-xtr[0,j])**2)

#print kte.shape
for i in range(0,idtr):
    for j in range(i,idtr):
        kte_train[i,j]=exp(-0.5*(xtr[0,i]-xtr[0,j])**2)
        kte_train[j,i]=kte_train[i,j]

### testing data covariance kernel  matrix
for i in range(0,idtr_l):
    for j in range(0,idtr_l):
        kte_test[i,j]=exp(-0.5*(xte[0,i]-xte[0,j])**2)

for i in range(0,idtr_l):
    for j in range(0,idtr_l):
        kte_kte[i,j]=exp(-0.5*(xtr[0,i]-xte[0,j])**2)

        #ktr[j,i]=ktr[i,j]
#print ktr.shape
#print kte.shape
kte_final=np.matrix(kte)
ktr_final=np.matrix(ktr)
kte_train_final=np.matrix(kte_train)
kte_test_final=np.matrix(kte_test)
kte_kte_final=np.matrix(kte_kte)
#print kte_final*ktr_final
### predict y
#print xte.tolist()
'''with noise'''
y_predict=kte_final*inv(ktr_final+0.7*np.eye(idtr,idtr))*ytr
y_predict_train=kte_train_final*inv(ktr_final+0.7*np.eye(idtr,idtr))*ytr

''' noise free'''
#y_predict=kte_final*inv(ktr_final)*ytr
#y_predict_train=kte_train_final*inv(ktr_final)*ytr

### covariance calculation
cov=kte_test_final-kte_final*inv(ktr_final+0.7*np.eye(idtr,idtr))*kte_kte_final
print cov.shape

#print len(xtest)
#print len(y_predict.tolist())
#plt.plot(xall,yall,'g',xtest,y_predict.tolist(),'')
plt.plot(xall,yall,'r',xtrain,y_predict_train.tolist(),'b+',xtest,y_predict.tolist(),'gs')
plt.show()
#A=inv(xtr*xtr.T+np.eye(len(xtr)))
#b=np.eye(len(xtr),len(xtr))
#A=inv(xtr*xtr.T+b)
#print b.shape
#print A.shape
#meanw=A*xtr*ytr
#print A.shape
#print meanw
