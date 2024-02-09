#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxopt
from numpy import linalg
from sklearn.model_selection import train_test_split
import os
import time

# Load the dataset
trainset = pd.read_csv('data/Xtr.csv', header=None)  # Assuming there's no header row
trainset = trainset.iloc[:,:-1]
train_labels = pd.read_csv('data/Ytr.csv').iloc[:, 1].values

# For testset
testset = pd.read_csv('data/Xte.csv', header=None)
testset = testset.iloc[:,:-1]

params = {'C': 1.0,
          'sigma': 5.0,
          'p': 3,
        }

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=params['p']):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=params['sigma']):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

params['kernel'] = gaussian_kernel

# Function to convert a row to an image
def row_to_image(row):
    # Split the row into three parts, for R, G, and B channels
    R = row[:1024].reshape(32, 32)
    G = row[1024:2048].reshape(32, 32)
    B = row[2048:].reshape(32, 32)
    # Stack the channels along the last dimension to form an RGB image
    image = np.stack([R, G, B], axis=-1)
    return image

# Function to calculate the Histogram of Oriented Gradients (HOG) for an image
def calculate_hog(image):
    # Convert the image to grayscale
    grayscale_image = np.dot(image, [0.2989, 0.5870, 0.1140])
    
    # Calculate the gradients in the x and y directions
    gradient_x = np.gradient(grayscale_image, axis=0)
    gradient_y = np.gradient(grayscale_image, axis=1)
    
    # Calculate the magnitude and direction of the gradients
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    direction = np.arctan2(gradient_y, gradient_x)
    
    # Define the number of bins for the histogram
    num_bins = 9
    
    # Calculate the histogram of oriented gradients
    histogram = np.zeros(num_bins)
    bin_width = 2 * np.pi / num_bins
    for i in range(grayscale_image.shape[0]):
        for j in range(grayscale_image.shape[1]):
            angle = direction[i, j]
            weight = magnitude[i, j]
            bin_index = int(angle / bin_width)
            histogram[bin_index] += weight
    
    return histogram

class BinarySVM(object):
    def __init__(self, kernel=linear_kernel, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1,n_samples),'d')
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        a = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print("%d support vectors out of %d points" % (len(self.a), n_samples))

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def _decision_function(self, X):
        if self.w is not None:
            decision = np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            decision = y_predict + self.b

        return np.sign(decision)
    

# Apply HOG to trainset
train_images = np.array([row_to_image(row.values) / 255.0 for index, row in trainset.iterrows()])
train_hog_features = np.array([calculate_hog(image) for image in train_images])

# For testset
test_images = np.array([row_to_image(row.values) / 255.0 for index, row in testset.iterrows()])
test_hog_features = np.array([calculate_hog(image) for image in test_images])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_hog_features, train_labels, test_size=0.1, random_state=42)

# Create a dictionary to store the trained models
models = {}

# Train a BinarySVM model for each class
for class_label in np.unique(train_labels):
    # Create a binary label vector for the current class
    binary_labels = np.where(y_train == class_label, 1, -1)
    
    # Create a BinarySVM model
    model = BinarySVM(kernel=params['kernel'], C=params['C'])
    
    # Fit the model on the training data
    model.fit(X_train, binary_labels)
    
    # Store the trained model in the dictionary
    models[class_label] = model

# Evaluate the models on the validation set
predictions = []
for features in X_val:
    # Predict the class label for each model
    class_scores = [model._decision_function([features])[0] for model in models.values()]
    
    # Select the class label with the highest score
    predicted_class = max(models.keys(), key=lambda x: class_scores[x])
    
    # Append the predicted class label to the list of predictions
    predictions.append(predicted_class)

# Calculate the accuracy of the OneVSAll model
accuracy = np.mean(predictions == y_val)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Predict the test set

# Create a dictionary to store the trained models
models = {}

# Train a BinarySVM model for each class
for class_label in np.unique(train_labels):
    # Create a binary label vector for the current class
    binary_labels = np.where(train_labels == class_label, 1, -1)
    
    # Create a BinarySVM model
    model = BinarySVM(kernel=params['kernel'], C=params['C'])
    
    # Fit the model on the training data
    model.fit(train_hog_features, binary_labels)
    
    # Store the trained model in the dictionary
    models[class_label] = model

test_predictions = []
for features in test_hog_features:
    # Predict the class label for each model
    class_scores = [model._decision_function([features])[0] for model in models.values()]
    
    # Select the class label with the highest score
    predicted_class = max(models.keys(), key=lambda x: class_scores[x])
    
    # Append the predicted class label to the list of predictions
    test_predictions.append(predicted_class)

test_predictions_df = pd.DataFrame({'Prediction' : test_predictions})
test_predictions_df.index += 1
test_predictions_df

# Save the predictions
timestamp = int(time.time())
submission_filename = f"submission{timestamp}.csv"
submission_filepath = os.path.join("submissions", submission_filename)
test_predictions_df.to_csv(submission_filepath, index_label='Id')

# Print the path to the submission file
print("Submission saved at:", submission_filepath)


# %%
