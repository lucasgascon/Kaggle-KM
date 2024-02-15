import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxopt
from numpy import linalg
from sklearn.model_selection import train_test_split
from kernels import linear_kernel, polynomial_kernel, gaussian_kernel
import os
import time
from tqdm import tqdm




class SVM_Dual:

    def __init__(self, C = 1, kernel=linear_kernel, epochs=1000, learning_rate= 0.001):
        self.alpha = None
        self.b = 0
        self.C = C
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.kernel = kernel
        self.K = None
        self.X = None
        self.y = None

    def fit(self,X,y):
        self.X = X
        self.y = y
        n_samples, n_features = X.shape
        self.alpha = np.random.random(n_samples) # n samples
        self.b = 0
        self.ones = np.ones(n_samples) 
        
        self.K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                self.K[i,j] = self.kernel(X[i,:], X[j,:])
        y_mul_kernal = np.outer(y, y) * self.K # yi yj K(xi, xj)

        for i in tqdm(range(self.epochs)):
            gradient = self.ones - y_mul_kernal.dot(self.alpha) # 1 – yk ∑ αj yj K(xj, xk)
            self.alpha += self.learning_rate * gradient # α = α + η*(1 – yk ∑ αj yj K(xj, xk)) to maximize
            self.alpha[self.alpha > self.C] = self.C # 0<α<C
            self.alpha[self.alpha < 0] = 0 # 0<α<C

            loss = np.sum(self.alpha) - 0.5 * np.sum(np.outer(self.alpha, self.alpha) * y_mul_kernal) # ∑αi – (1/2) ∑i ∑j αi αj yi yj K(xi, xj)
            
        alpha_index = np.where((self.alpha) > 0 & (self.alpha < self.C))[0]
        
        # for intercept b, we will only consider α which are 0<α<C 
        b_list = []        
        for index in alpha_index:
            
            b_list.append(y[index] - (self.alpha * y).dot(self.K[:,index]))

        self.b = np.mean(b_list) # avgC≤αi≤0{ yi – ∑αjyj K(xj, xi) }
            
    def predict(self, X):
        return np.sign(self._decision_function(X))
    
    def score(self, X, y):
        y_hat = self.predict(X)
        return np.mean(y == y_hat)
    
    def _decision_function(self, X):
        n_samples, n_features = self.X.shape
        temp_vect = np.zeros(n_samples)
        for i in range(n_samples):
            temp_vect[i] = self.kernel(self.X[i,:], X)
        return (self.alpha * self.y).dot(temp_vect) + self.b
