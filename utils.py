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

def viz_image(image, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(image)
    return ax