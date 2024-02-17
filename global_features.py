import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxopt
from numpy import linalg
from sklearn.model_selection import train_test_split
import os
import time

class kernelPCA():
    def __init__(self,kernel, n_components = -1, plt = False):                             
        self.kernel = kernel          # <---
        self.alpha = None # Matrix of shape N times d representing the d eingenvectors alpha corresp
        self.support = None # Data points where the features are evaluated
        self.n_components = n_components ## Number of principal components
        self.lmbd = None # Eigenvalues
        
    def fit(self,X):
        n_samples, n_features = X.shape
        self.support = X
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])
            
        # Center the kernel matrix
        N = K.shape[0]
        one_n = np.ones((N,N)) / N
        K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    
        # Obtaining eigenpairs from the centered kernel matrix
        eigvals, eigvecs = linalg.eigh(K)
        
        if self.n_components < 0:
            self.n_components = 0
            ratio = 0
            while ratio < 0.8:
                self.n_components += 1
                ratio = sum(eigvals[-i] for i in range(1,self.n_components+1))/sum(eigvals)


            
        eig_pc = np.column_stack((eigvecs[:,-i]/np.sqrt(eigvals[-i]) for i in range(1,self.n_components+1)))
        self.alpha = eig_pc # Matrix of shape N times d representing the d eingenvectors alpha corresp
        X_pca = np.zeros(((n_samples, self.n_components)))
        for i in range(self.n_components):
            X_pca[:,i] = K@eig_pc[:,i]
        self.lmbd = eigvals
        return X_pca
    def transform(self,x):
        n_samples, n_features = x.shape
        X_pca = np.zeros(((n_samples, self.n_components)))
        
        K = np.zeros((n_samples, self.support.shape[0]))
        for i in range(n_samples):
            for j in range(self.support.shape[0]):
                K[i,j] = self.kernel(x[i], self.support[j])
                
        for i in range(self.n_components):
            X_pca[:,i] = K@self.alpha[:,i]
            
        if not plt:
            return X_pca
        else:
            explained_var = [sum(self.lmbd[-i] for i in range(1,i+1))/sum(self.lmbd) for i in range(1, self.n_components+1)]
            plt.figure()
            plt.plot(np.arange(1, len(self.lmbd)+1),explained_var)
            plt.xlabel('Number of components')
            plt.ylabel('Explained variance')
            plt.title('Explained variance vs Number of components taken in the PCA for kernel ', kernel.__name__)
            return X_pca


# To check 
def compute_fishervector(X, means, covariances, priors):
    n_samples, n_features = X.shape
    n_components, _ = means.shape
    n_features = n_features // n_components
    
    FV = np.zeros((n_components*2*n_features))
    
    for i in range(n_components):
        diff = X - means[i]
        cov = covariances[i]
        cov_inv = np.linalg.inv(cov)
        diff = diff.T
        diff = diff.T
        diff_cov_inv = diff@cov_inv
        FV[i*n_features:i*n_features+n_features] = np.sum(diff_cov_inv, axis=0)
        FV[n_components*n_features+i*n_features:n_components*n_features+i*n_features+n_features] = np.sum(diff_cov_inv**2 - 1, axis=0)
    
    FV = FV / np.sqrt(np.sum(FV**2))
    return FV
 