import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxopt
from numpy import linalg
from sklearn.model_selection import train_test_split
import os
import time

def compute_kernelPCA(X, kernel, n_components = -1):
    n_samples, n_features = X.shape
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i,j] = kernel(X[i], X[j])
            
    # Center the kernel matrix
    N = K.shape[0]
    one_n = np.ones((N,N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    
    # Obtaining eigenpairs from the centered kernel matrix
    eigvals, eigvecs = linalg.eigh(K)
    
    if n_components < 0:
        n_components = 0
        ratio = 0
        while ratio < 0.8:
            n_components += 1
            ratio = sum(eigvals[-i] for i in range(1,n_components+1))/sum(eigvals)
            
    eig_pc = np.column_stack((eigvecs[:,-i]/np.sqrt(eigvals[-i]) for i in range(1,n_components+1)))
    X_pca = np.zeros(((n_samples, n_components)))
    for i in range(n_components):
        X_pca[:,i] = K@eig_pc[:,i]
    
    return X_pca
 