import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxopt
from numpy import linalg
from sklearn.model_selection import train_test_split
from train import params
import os
import time

def linear_kernel(x, y):
    return np.dot(x, y)

def polynomial_kernel(x, y, p=params['p']):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=params['sigma']):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

def sigmoid_kernel(x, y, gamma = params['gamma'], r = params['r']):
    return np.tanh(gamma*np.dot(x, y) + r)

def laplacian_kernel(x, y, sigma = params['sigma']):
    return np.exp(-linalg.norm(x-y, ord=1) / sigma**2)