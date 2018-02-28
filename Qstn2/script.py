# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 12:14:13 2018

@author: HP
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVR

def preprocessing(filepath):
    df = pd.read_csv(filepath)
    
    fil = savgol_filter
    transformed = fil(df[df.columns[1:3579]],11,8)
    df.drop(df.columns[1:3579],axis=1,inplace=True)
        
    lb = LabelBinarizer()
    df['Depth'] = lb.fit_transform(df['Depth'])
    index = df['PIDN']
    
    if 'train' in filepath:
        y = df[['Ca','P','pH','SOC','Sand']].values
        df.drop(['Ca','P','pH','SOC','Sand'],axis=1,inplace=True)
    X = np.concatenate((df[df.columns[1:]].values,transformed),axis=1)
    
    if 'train' in filepath:
        return X,y,index
    else:
        return X,index

#reg = MLPRegressor(hidden_layer_sizes=(100,),verbose=True,max_iter=200)
reg = SVR(C=10000.0)

X_train,y_train,_ = preprocessing('train.csv')
X_test,index = preprocessing('test.csv')

preds = np.zeros((X_test.shape[0], 5))
for i in range(5):
    reg.fit(X_train,y_train[:,i])
    preds[:,i] = reg.predict(X_test)
    
sol = pd.DataFrame(preds,index,columns=['Ca','P','pH','SOC','Sand'])

sol.to_csv('solution.csv',index_label=['PIDN'])