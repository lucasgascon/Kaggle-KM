import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxopt
from numpy import linalg
from sklearn.model_selection import train_test_split
import os
import time




# Function to convert a row to an image
def row_to_image(row):
    # Split the row into three parts, for R, G, and B channels
    R = row[:1024].reshape(32, 32)
    G = row[1024:2048].reshape(32, 32)
    B = row[2048:].reshape(32, 32)
    # Stack the channels along the last dimension to form an RGB image
    image = np.stack([R, G, B], axis=-1)
    return image

def viz_image(image, ax=None, exp_norm = True):
    new_im = np.zeros(image.shape)
    for i in range(3):
        if exp_norm:
            new_im[:,:,i] = 1/ (1 + np.exp(-10*image[:,:,i]))
        else:
            new_im[:,:,i] = (image[:,:,i] - np.min(image[:,:,i]) )/ (np.max(image[:,:,i]) -  np.min(image[:,:,i]))
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(new_im)
    return ax