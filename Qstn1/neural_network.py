# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 13:29:25 2018

@author: HP
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
np.random.seed(1)

def sigmoid(z):
    return 1/(1 + np.exp(-z))
    
def sigmoid_backward(z):
    return sigmoid(z)*(1 - sigmoid(z))
    
def layer_sizes(X,y):
    n_x = X.shape[0]
    n_h1 = 10
    n_h2 = 5
    n_y = y.shape[0]
    
    return (n_x, n_h1, n_h2, n_y)
    
def initialize_parameters(n_x, n_h1, n_h2, n_y):
    
    W1 = np.random.randn(n_h1, n_x) * 0.01
    b1 = np.zeros(shape=(n_h1, 1))
    W2 = np.random.randn(n_h2, n_h1) * 0.01
    b2 = np.zeros(shape=(n_h2, 1))
    W3 = np.random.randn(n_y, n_h2) * 0.01
    b3 = np.zeros(shape=(n_y, 1))
    
    parameters = [W1, b1, W2, b2, W3, b3]
    
    return parameters

def forward_propagation(X, parameters):
    
    W1, b1, W2, b2, W3, b3 = parameters
   
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = np.tanh(Z2)
    Z3 = np.dot(W3,A2) + b3
    A3 = sigmoid(Z3)
    
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "A3": A3}
    
    return A3, cache
    
def compute_cost(A3, Y, parameters,lambd = 0):
    
    m = Y.shape[1]
    W1, b1, W2, b2, W3, b3 = parameters
    
    cost = - np.sum(np.multiply(np.log(A3), Y) + np.multiply((1 - Y), np.log(1 - A3))) / m + (lambd/(2*m))*(sum(sum(W1**2))+sum(sum(W2**2))+(sum(sum(W3**2))))
    cost = np.squeeze(cost)     
    
    return cost
    
def backward_propagation(parameters, cache, X, Y):
    
    m = X.shape[1]
    
    W1 = parameters[0]
    W2 = parameters[2]
    W3 = parameters[4]
    
    A1 = cache['A1']
    A2 = cache['A2']
    A3 = cache['A3']
    
    dZ3 = A3 - Y
    dW3 = (1 / m) * np.dot(dZ3, A2.T)
    db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)
    dZ2 = np.multiply(np.dot(W3.T, dZ3), 1 - np.power(A2, 2))
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    
    grads = [dW1, db1, dW2, db2, dW3, db3]
    
    return grads
    
def update_parameters(parameters, grads, learning_rate=1.2):
    
    W1, b1, W2, b2, W3, b3 = parameters
    dW1, db1, dW2, db2, dW3, db3 = grads
    
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    W3 = W3 - learning_rate * dW3
    b3 = b3 - learning_rate * db3
    
    parameters = [W1, b1, W2, b2, W3, b3]
    
    return parameters
    
def nn_model(X, Y, n_h1, n_h2, epsilon=1e-7, num_iterations=10000, print_cost=False):
    
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[3]
    temp_cost = 1000
    
    parameters = initialize_parameters(n_x, n_h1, n_h2, n_y)

    for i in range(0, num_iterations):
         
        A3, cache = forward_propagation(X, parameters)  
        cost = compute_cost(A3, Y, parameters, lambd=1.4)
        if temp_cost - cost < epsilon:
            break
        grads = backward_propagation(parameters, cache, X, Y) 
        parameters = update_parameters(parameters, grads)
        
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
            temp_cost = cost

    return parameters
    
def predict(parameters, X):

    A3, cache = forward_propagation(X, parameters)
    predictions = np.round(A3)
    
    return predictions
    
'''This neural network with two hidden layers is being trained on the MNIST dataset below.'''


train = pd.read_csv('train.csv')

lb = LabelBinarizer()

y = train.label.values
y = y[:10000]
y = lb.fit_transform(y)
 
X = train[train.columns[1:]].values
X = X[:10000,:]
   
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)

params = nn_model(X_train.T, y_train.T, 20, 20, epsilon=1e-7, print_cost=True)

pred = predict(params, X_test.T)