import numpy as np
import cvxopt
from kernels import linear_kernel, polynomial_kernel, gaussian_kernel
import time


class SVM_SGD:

    def __init__(self, C = 1, kernel=linear_kernel, epochs=1000, learning_rate= 0.001):
        self.alpha = None
        self.b = 0
        self.C = C
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.kernel = kernel
        self.K = None
        self.X = None
        self.y = None

    def fit(self,X,y):
        self.X = X
        self.y = y
        n_samples, n_features = X.shape
        self.alpha = np.random.random(n_samples) # n samples
        self.b = 0
        self.ones = np.ones(n_samples) 
        
        self.K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                self.K[i,j] = self.kernel(X[i,:], X[j,:])
        y_mul_kernal = np.outer(y, y) * self.K # yi yj K(xi, xj)

        

        for i in range(self.epochs): 
            gradient = - self.ones + y_mul_kernal.dot(self.alpha) # 1 – yk ∑ αj yj K(xj, xk)
            self.alpha -= self.learning_rate * gradient # α = α + η*(1 – yk ∑ αj yj K(xj, xk)) to maximize
            self.alpha[self.alpha > self.C] = self.C # 0<α<C
            self.alpha[self.alpha < 0] = 0 # 0<α<C

            loss = - np.sum(self.alpha) + 0.5 * np.sum(np.outer(self.alpha, self.alpha) * y_mul_kernal) # ∑αi – (1/2) ∑i ∑j αi αj yi yj K(xi, xj)
            
        alpha_index = np.where((self.alpha) > 0 & (self.alpha < self.C))[0]
        self.support_vectors = X[alpha_index,:]
        self.alpha_y = self.alpha[alpha_index] * y[alpha_index]
        
        # for intercept b, we will only consider α which are 0<α<C 
        b_list = []        
        for index in alpha_index:
            
            b_list.append(y[index] - (self.alpha * y).dot(self.K[:,index]))

        self.b = np.mean(b_list) # avgC≤αi≤0{ yi – ∑αjyj K(xj, xi) }
            
    def predict(self, X):
        return self._decision_function(X) 
    
    def score(self, X, y):
        y_hat = self.predict(X)
        return np.mean(y == y_hat)
    
    def _decision_function(self, X):
        n_samples, n_features = self.support_vectors.shape
        temp_vect = np.zeros(n_samples)
        for i in range(n_samples):
            temp_vect[i] = self.kernel(self.support_vectors[i,:], X)
        d =  (self.alpha_y).dot(temp_vect) + self.b    
        return 2 * (d+self.b> 0) - 1


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