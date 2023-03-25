# -*- coding: utf-8 -*-
"""
Created on Sat May 28 10:37:22 2022

@author: Shayal
"""
import numpy as np
from scipy import io
from statistics import stdev
from statistics import mean
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix

from scipy import stats




def feature_extr(X,Y):
    pix_avg=np.zeros((len(X),1),float)
    std_pixel=np.zeros((len(X),1),float)

    for i in range(len(X)):
    
        std_pixel[i][0]=stdev(X[i,:])    
        pix_avg[i][0]=mean(X[i,:])

    mod_data=np.concatenate((pix_avg,std_pixel,Y),axis=1)
    mod_data=pd.DataFrame(mod_data,columns=['F1','F2','class'])
    return mod_data 


data=io.loadmat(r"F:\sml\PROJ 1\mnist_data.mat")
X_train=data["trX"]
Y_train=data["trY"]
X_test=data["tsX"]
Y_test=data["tsY"]
Y_train=Y_train.transpose()
Y_test=Y_test.transpose()
y_pred=np.zeros((len(Y_test),1))

train_df=feature_extr(X_train, Y_train)
test_df=feature_extr(X_test, Y_test)

#Naive Bayes Classifier


mean_features=train_df.groupby('class')['F1','F2'].mean()
std_features=train_df.groupby('class')['F1','F2'].std()

#digit7
m_f1_0=mean_features.iloc[0][0]
m_f2_0=mean_features.iloc[0][1]
std_f1_0=std_features.iloc[0][0]
std_f2_0=std_features.iloc[0][1]

#digit8
m_f1_1=mean_features.iloc[1][0]
m_f2_1=mean_features.iloc[1][1]
std_f1_1=std_features.iloc[1][0]
std_f2_1=std_features.iloc[1][1]

#prior probability of class
prob_7=np.count_nonzero(Y_train==0)/len(Y_train)
prob_8=1-prob_7

#posterior probaility p(x|y=0) for train data
data_7=train_df.loc[train_df['class']==0]
post_prob7=stats.norm.pdf(data_7.iloc[:,0],m_f1_0, std_f1_0)*stats.norm.pdf(data_7.iloc[:,1],m_f2_0, std_f2_0)


#posterior probaility p(x|y=1) for train data
data_8=train_df.loc[train_df['class']==1]
post_prob8=stats.norm.pdf(data_8.iloc[:,0],m_f1_1, std_f1_1)*stats.norm.pdf(data_8.iloc[:,1],m_f2_1, std_f2_1)

#p(x|y=digit7) for test data
post_prob_test7=stats.norm.pdf(test_df.iloc[:,0],m_f1_0, std_f1_0)*stats.norm.pdf(test_df.iloc[:,1],m_f2_0, std_f2_0)
#p(x|y=digit8) for test data
post_prob_test8=stats.norm.pdf(test_df.iloc[:,0],m_f1_1, std_f1_0)*stats.norm.pdf(test_df.iloc[:,1],m_f2_1, std_f2_0)

for i in range(len(X_test)):
    if (prob_7*post_prob_test7[i])>(prob_8*post_prob_test8[i]):
        y_pred[i][0]=0
    else:
        y_pred[i][0]=1
    

con_mat=confusion_matrix(Y_test,y_pred)   
print('accuracy of predicting digit 7:',con_mat[0][0]/sum(con_mat)[0])    
print('accuracy of predicting digit 8:',con_mat[1][1]/sum(con_mat)[1])    
    
print("total Naive Bayes Classifier accuracy is:",accuracy_score(Y_test, y_pred)*100)   

#logistic regression
