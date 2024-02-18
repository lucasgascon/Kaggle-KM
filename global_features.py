import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg


class kernelPCA():
    
    def __init__(self,kernel, n_components = -1, plt = False): 
                                    
        self.kernel = kernel          # <---
        self.alpha = None # Matrix of shape N times d representing the d eingenvectors alpha corresp
        self.support = None # Data points where the features are evaluated
        self.n_components = n_components ## Number of principal components
        self.lmbd = None # Eigenvalues
        self.plt = plt
        
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


            
        eig_pc = np.column_stack([eigvecs[:,-i]/np.sqrt(eigvals[-i]) for i in range(1,self.n_components+1)])
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
            
        if not self.plt:
            return X_pca
        else:
            explained_var = [sum(self.lmbd[-i] for i in range(1,i+1))/sum(self.lmbd) for i in range(1, len(self.lmbd)+1)]
            plt.figure()
            plt.plot(np.arange(1, len(self.lmbd)+1),explained_var)
            plt.xlabel('Number of components')
            plt.ylabel('Explained variance')
            plt.title('Explained variance vs Number of components taken in the PCA for kernel ', kernel.__name__)
            return X_pca


class Kmeans():
    def __init__(self, k, max_iter = 100, eps = 1e-8):
        self.k = k
        self.max_iter = max_iter
        self.eps = eps
        self.clusters = None
        self.centroids = None
        
    def fit(self, X):
        # X is a list of arrays, each array is a set of local descriptors for an image
        stacked_descriptors = np.vstack(np.array(X))
        n_descriptors = stacked_descriptors.shape[0]
        # Randomly initialize the centroids
        centroids = stacked_descriptors[np.random.choice(n_descriptors, self.k, replace = False),:]
        prev_centroids = centroids.copy()
        for i in range(self.max_iter):
            # Assign each sample to the nearest centroid
            distances = np.zeros((n_descriptors, self.k))
            for j in range(self.k):
                distances[:,j] = np.linalg.norm(stacked_descriptors - centroids[j], axis = 1)
            clusters = np.argmin(distances, axis = 1)
            # Update the centroids
            for j in range(self.k):
                centroids[j] = np.mean(stacked_descriptors[clusters == j,:], axis = 0)
            # Check for convergence
            if np.linalg.norm(centroids - prev_centroids) < self.eps:
                break
            prev_centroids = centroids.copy()
            self.clusters = clusters
            self.centroids = centroids
            
    def predict(self, X):
        # X is a list of arrays, each array is a set of local descriptors for an image
        stacked_descriptors = np.vstack(np.array(X))
        n_descriptors = stacked_descriptors.shape[0]
        distances = np.zeros((n_descriptors, self.k))
        for j in range(self.k):
            distances[:,j] = np.linalg.norm(stacked_descriptors - self.centroids[j], axis = 1)
        clusters = np.argmin(distances, axis = 1)
        return clusters
  

# Bag of words
class BoW():
    def __init__(self, k, max_iter = 100, eps = 1e-8):
        self.k = k
        self.max_iter = max_iter
        self.eps = eps
        self.kmeans = None
        
    def fit(self, local_descriptors_list):
        n_images = len(local_descriptors_list)
        train_im_features = np.zeros((n_images, self.k))
        kmeans = Kmeans(self.k, self.max_iter, self.eps)
        kmeans.fit(local_descriptors_list)
        self.kmeans = kmeans
        for i in range(n_images):
            for j in range(local_descriptors_list[i].shape[0]):
                train_im_features[i, kmeans.clusters[j]] += 1
        self.kmeans = kmeans
        return train_im_features
    
    def predict(self, local_descriptors_list):
        n_images = len(local_descriptors_list)
        im_features = np.zeros((n_images, self.k))
        clusters = self.kmeans.predict(local_descriptors_list)
        for i in range(n_images):
            for j in range(local_descriptors_list[i].shape[0]):
                im_features[i, clusters[j]] += 1
        return im_features
     
"""
À VÉRIFIER   
def FisherVectorEncoder(local_descriptors_list, gmm, n_components):
    n_images = len(local_descriptors_list)
    n_gaussians = gmm.means_.shape[0]
    n_features = n_gaussians * n_components
    fisher_vector = np.zeros((n_images, n_features))
    for i in range(n_images):
        local_descriptors = local_descriptors_list[i]
        n_descriptors = local_descriptors.shape[0]
        # Compute the posterior probabilities
        post_prob = gmm.predict_proba(local_descriptors)
        # Compute the Fisher vector
        fisher_vector[i] = np.zeros((n_features))
        for j in range(n_gaussians):
            # Compute the gradients with respect to the means
            grad_means = np.sum(post_prob[:,j][:,np.newaxis] * (local_descriptors - gmm.means_[j]), axis = 0)
            # Compute the gradients with respect to the covariances
            grad_covs = np.sum(post_prob[:,j][:,np.newaxis] * ((local_descriptors - gmm.means_[j])**2 - gmm.covariances_[j]), axis = 0)
            # Compute the gradients with respect to the weights
            grad_weights = np.sum(post_prob[:,j])
            fisher_vector[i][j*n_components:(j+1)*n_components] = np.concatenate((grad_means, grad_covs, grad_weights))
        # L2 normalize the Fisher vector
        fisher_vector[i] /= np.sqrt(np.sum(fisher_vector[i]**2))
    return fisher_vector
"""
