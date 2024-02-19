import numpy as np
from tqdm import tqdm

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

class GaussianKernelForAngle:
    def __init__(self, sigma):
        self.sigma = sigma
        self.name = f'gaussian_angle_{sigma:.5f}'

    def calc(self, x, y):
        # Calculation using trigonometric functions for angles
        aux = (np.sin(x) - np.sin(y)) ** 2 + (np.cos(x) - np.cos(y)) ** 2
        return np.exp(-aux / (2 * self.sigma ** 2))

    def build_K(self, X, Y=None):
        if Y is None:
            Y = X
        # Transform X and Y to include both sine and cosine values
        X2 = np.concatenate((np.sin(X), np.cos(X)), axis=1)
        Y2 = np.concatenate((np.sin(Y), np.cos(Y)), axis=1)

        # Utilize broadcasting to compute the squared differences without explicit loops
        # Expand X2 and Y2 to 3D arrays to enable broadcasting
        X2_expanded = X2[:, np.newaxis, :]
        Y2_expanded = Y2[np.newaxis, :, :]

        # Compute the squared L2 norm (squared Euclidean distance) between all pairs of points
        squared_diff = np.sum((X2_expanded - Y2_expanded) ** 2, axis=2)

        # Apply the Gaussian kernel formula
        K = np.exp(-squared_diff / (2 * self.sigma ** 2))

        return K