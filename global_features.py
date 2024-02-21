import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg
from tqdm import trange, tqdm
from scipy.special import logsumexp


class kernelPCA():
    
    def __init__(self,kernel, n_components = -1, plt = False, with_norm = False): 
                                    
        self.kernel = kernel          # <---
        self.alpha = None # Matrix of shape N times d representing the d eingenvectors alpha corresp
        self.support = None # Data points where the features are evaluated
        self.n_components = n_components ## Number of principal components
        self.lmbd = None # Eigenvalues
        self.plt = plt
        self.with_norm = with_norm
        
    def fit(self,X):
        
        n_samples, n_features = X.shape
        self.support = X
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])
            
        # Center the kernel matrix
        N = K.shape[0]
        one_N = np.ones((N,N)) / N
        if self.with_norm:
            Kcentered = (np.eye(N) - one_N)@(K)@(np.eye(N) - one_N)
        else:
            Kcentered = K
        
        # Obtaining eigenpairs from the centered kernel matrix
        eigvals, eigvecs = np.linalg.eigh(Kcentered)
        
        self.lmbda = eigvals[::-1] # Descending order
        self.alpha = eigvecs[:,::-1]
       
        
        if self.n_components < 0:
            self.n_components = 0
            ratio = 0
            while ratio < 0.8:
                self.n_components += 1
                ratio = sum(self.lmbda[i] for i in range(0,self.n_components))/sum(self.lmbda)



        return  K@self.alpha[:,:self.n_components]
    
    def transform(self,x):
        
        n_samples, n_features = x.shape
        n = self.support.shape[0]

        K = np.zeros((n_samples, n))
        for i in range(n_samples):
            for j in range(n):
                K[i,j] = self.kernel(x[i], self.support[j])
        
        one_N =  np.ones((n_samples,n_samples))/n_samples

        one_n =  np.ones((n,n))/n
        if self.with_norm:
            K= (np.eye(n_samples) - one_N)@(K)@(np.eye(n) - one_n)
            
        if not self.plt:
            return  K@self.alpha[:,:self.n_components]
        else:
            explained_var = [sum(self.lmbd[-i] for i in range(1,i+1))/sum(self.lmbd) for i in range(1, len(self.lmbd)+1)]
            plt.figure()
            plt.plot(np.arange(1, len(self.lmbd)+1),explained_var)
            plt.xlabel('Number of components')
            plt.ylabel('Explained variance')
            plt.title('Explained variance vs Number of components taken in the PCA for kernel ', self.kernel.__name__)
            return  K@self.alpha[:,:self.n_components]


class Kmeans():
    def __init__(self, k, max_iter = 100, eps = 1e-8):
        self.k = k
        self.max_iter = max_iter
        self.eps = eps
        self.clusters = None
        self.centroids = None
        
    def fit(self, X, already_stacked = False):
        # X is a list of arrays, each array is a set of local descriptors for an image
        
        if already_stacked:
            stacked_descriptors = X
        else:
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
Implementation of the Fisher Vector algorithm for image classification
"""    

def EM_algorithm(local_descriptors_list, n_gaussians, n_iter = 100, starting_lmbda = None, eps = 0.01):
    
    l = np.zeros(n_iter) # Store log-likelihood at each step
    x = np.array(local_descriptors_list[0])
    for i in range(1,len(local_descriptors_list)):
        if local_descriptors_list[i] is None:
            continue
        else:
            x = np.vstack((x, local_descriptors_list[i]))
    m = n_gaussians
    d = x.shape[1]
    n = x.shape[0]
    
    #initialize params
    if starting_lmbda is None:
        kmeans = Kmeans(m)
        kmeans.fit(x, already_stacked = True)
        mu_t = kmeans.centroids
        Sigma_t = np.ones((m,d))
        alpha_t = np.ones(m)
        alpha_t /= alpha_t.sum() 
    else:
        mu_t, Sigma_t, alpha_t = starting_lmbda
        
    print('EM algorithm started')
    for t in trange(n_iter):
        
        log_tau_t = np.zeros((n, m))
            
        # STEP E
        for gauss_ind in range(m):
            sign, logdet = np.linalg.slogdet(np.diag(Sigma_t[gauss_ind,:]))
            inv_Sigma_t = np.diag(1/(Sigma_t[gauss_ind,:]+1e-6))
            arg_exp = np.array([- (x[i,:]-mu_t[gauss_ind,:]).T @ inv_Sigma_t @ (x[i,:]-mu_t[gauss_ind,:]) / 2 for i in range(n) ])
            log_tau_t[:, gauss_ind] = np.log(alpha_t[gauss_ind]) - 0.5 * sign * logdet - d * np.log(2*np.pi)/2  + arg_exp

        tau_t = np.exp(log_tau_t - logsumexp(log_tau_t, axis=1).reshape(n, 1)) #normalize tau_t 


          
        # STEP M 
        sum_tau_t = np.sum(tau_t, axis = 0)
        alpha_t = sum_tau_t/n
        
        for gauss_ind in range(m):
            mu_t[gauss_ind,:] = tau_t[:,gauss_ind].T@x/sum_tau_t[gauss_ind]
            Sigma_t[gauss_ind,:] =np.diag((tau_t[:,gauss_ind]*(x-mu_t[gauss_ind,:]).T@(x-mu_t[gauss_ind,:]))/sum_tau_t[gauss_ind]+ eps * np.eye(d,d))
        
        if not test(alpha_t, mu_t, tau_t, n, m, d):
            break

    return {'alpha':alpha_t, 'mu':mu_t, 'sigma':Sigma_t}



def test(alpha_t, mu_t, tau_t, n, m, d, eps = 1e-3):
    ok = True
    if alpha_t.shape != (m,):
        ok = False
        print("1 ", alpha_t.shape)
    if np.abs(alpha_t.sum() - 1) > eps:
        ok = False
        print("2 :", int(alpha_t.sum()))
    if mu_t.shape != (m,d):
        ok = False
        print("3")
    if tau_t.shape != (n,m):
        ok = False
        print("5")
    if np.abs(tau_t.sum(axis=1)[0] - 1) > eps:
        ok = False
        print("6 ",tau_t.sum(axis=1)[0])
    return ok

def multivariate_normal_density(X, mu, Sigma):
    d = X.shape[0]
    det = np.linalg.det(np.diag(Sigma))
    inv = np.diag(1 / (Sigma+1e-8))
    norm = 1 / ((2 * np.pi) ** (d / 2) * np.sqrt(det))
    exponent = -0.5 * np.sum(np.dot((X - mu), inv) * (X - mu))
    return norm * np.exp(exponent)


class FisherVectorGMM:
    def __init__(self, n_gaussians, n_iter = 100):
        self.n_gaussians = n_gaussians
        self.n_iter = n_iter
        self.gmm = None
        self.fisher_vector = None
        
    def fit(self, local_descriptors_list):
        # Fit a GMM to the local descriptors
        self.gmm = EM_algorithm(local_descriptors_list, self.n_gaussians)

    def transform(self, local_descriptors):
        # Compute the Fisher Vector for the given local descriptors
        n_gaussians = self.n_gaussians
        gmm = self.gmm
        T = local_descriptors.shape[0]
        S0 = np.zeros(n_gaussians)
        S1 = np.zeros((n_gaussians, local_descriptors.shape[1]))
        S2 = np.zeros((n_gaussians, local_descriptors.shape[1]))
        for t in range(T):
            x = local_descriptors[t,:]
            # compute u_lambda in the log domain
            logsum = gmm['alpha'][1]*multivariate_normal_density(x, gmm['mu'][1,:], gmm['sigma'][1,:])
            for i in range(1,n_gaussians):
                logsum += np.log(1+np.exp(logsum-np.log(gmm['alpha'][i] * multivariate_normal_density(x, gmm['mu'][i,:],gmm['sigma'][i,:]))))
            
            for k in range(n_gaussians):
                post_prob = np.exp(np.log(gmm['alpha'][k] * 
                                            multivariate_normal_density(x, gmm['mu'][k,:], gmm['sigma'][k,:]))-logsum)
                S0[k] += post_prob
                S1[k,:] += post_prob * x
                S2[k,:] += post_prob * (x ** 2)
                
        g_alpha = (S0 - T * gmm['alpha'])/(np.sqrt(gmm['alpha']))
        g_mu = (S1 - (gmm['mu'] * S0.reshape(-1,1)))/(np.sqrt(gmm['alpha']).reshape(-1,1)*gmm['sigma'])
        g_sigma = (S2 - 2 * gmm['mu'] * S1 + (gmm['mu'] ** 2 - gmm['sigma']**2)* S0.reshape(-1,1))/(np.sqrt(2 * gmm['alpha']).reshape(-1,1)*gmm['sigma']**2)
        fisher_vector = np.concatenate((g_alpha, g_mu.flatten(), g_sigma.flatten()))
        fisher_vector = np.sign(fisher_vector) * np.sqrt(np.abs(fisher_vector))
        fisher_vector /= np.sqrt(np.dot(fisher_vector, fisher_vector))
        return fisher_vector
