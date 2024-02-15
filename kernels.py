import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxopt
from numpy import linalg
from sklearn.model_selection import train_test_split
import os
import time




params = {'C': 1.0,
          'sigma': 5.0,
          'p': 3,
        }
def linear_kernel(x, y):
    return np.dot(x, y)

def polynomial_kernel(x, y, p=params['p']):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=params['sigma']):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))