import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxopt
from numpy import linalg
from sklearn.model_selection import train_test_split
import os
import time



def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=params['p']):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=params['sigma']):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))