import numpy as np

def linear_kernel(x, y):
    return np.dot(x, y)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=1.):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

def sigmoid_kernel(x, y, gamma = 1., r = 1.):
    return np.tanh(gamma*np.dot(x, y) + r)

def laplacian_kernel(x, y, sigma = 1.):
    return np.exp(-np.linalg.norm(x-y, ord=1) / sigma**2)

def chi2_kernel(x, y, sigma = 1.):
    return np.exp(-sigma*np.sum((x-y)**2 / (x+y+1e-6)))