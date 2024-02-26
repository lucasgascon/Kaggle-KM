# Description: This file contains the experiments to be run.


experiments = {
    # Hog + other features + linear SVM
    1: {'hog': True, 'raw': False, 'lbp': False, 'PCA': 0, 'kernelPCA': 'linear_kernel', 'strat': 'onevsone', 'SVM': 'CVXOPT', 'kernelSVM': 'linear_kernel', 'C': 1, 'sigma': 1, 'p': 1, 'subname': '1', 'gamma': 1, 'r': 1, 'bow': False, 'k': 1, 'sift': False, 'fishervect': False, 'with_norm': False}, 
    2: {'hog': True, 'raw': True, 'lbp': False, 'PCA': 0, 'kernelPCA': 'linear_kernel', 'strat': 'onevsone', 'SVM': 'CVXOPT', 'kernelSVM': 'linear_kernel', 'C': 1, 'sigma': 1, 'p': 1, 'subname': '1', 'gamma': 1, 'r': 1, 'bow': False, 'k': 1, 'sift': False, 'fishervect': False, 'with_norm': False},  
    3: {'hog': True, 'raw': False, 'lbp': True, 'PCA': 0, 'kernelPCA': 'linear_kernel', 'strat': 'onevsone', 'SVM': 'CVXOPT', 'kernelSVM': 'linear_kernel', 'C': 1, 'sigma': 1, 'p': 1, 'subname': '1', 'gamma': 1, 'r': 1, 'bow': False, 'k': 1, 'sift': False, 'fishervect': False, 'with_norm': False},
    4: {'hog': True, 'raw': False, 'lbp': False, 'PCA': 0, 'kernelPCA': 'linear_kernel', 'strat': 'onevsone', 'SVM': 'CVXOPT', 'kernelSVM': 'linear_kernel', 'C': 1, 'sigma': 1, 'p': 1, 'subname': '1', 'gamma': 1, 'r': 1, 'bow': False, 'k': 1, 'sift': True, 'fishervect': False, 'with_norm': False}, 
    5: {'hog': True, 'raw': True, 'lbp': True, 'PCA': 0, 'kernelPCA': 'linear_kernel', 'strat': 'onevsone', 'SVM': 'CVXOPT', 'kernelSVM': 'linear_kernel', 'C': 1, 'sigma': 1, 'p': 1, 'subname': '1', 'gamma': 1, 'r': 1, 'bow': False, 'k': 1, 'sift': True, 'fishervect': False, 'with_norm': False},
    # Bow with Sift + linear SVM + other features   
    6: {'hog': False, 'raw': True, 'lbp': False, 'PCA': 0, 'kernelPCA': 'linear_kernel', 'strat': 'onevsone', 'SVM': 'CVXOPT', 'kernelSVM': 'linear_kernel', 'C': 1, 'sigma': 1, 'p': 1, 'subname': '1', 'gamma': 1, 'r': 1, 'bow': True, 'k': 20, 'sift': True, 'fishervect': False, 'with_norm': False}, 
    7: {'hog': False, 'raw': False, 'lbp': True, 'PCA': 0, 'kernelPCA': 'linear_kernel', 'strat': 'onevsone', 'SVM': 'CVXOPT', 'kernelSVM': 'linear_kernel', 'C': 1, 'sigma': 1, 'p': 1, 'subname': '1', 'gamma': 1, 'r': 1, 'bow': True, 'k': 20, 'sift': True, 'fishervect': False, 'with_norm': False}, 
    # Hog gaussian kernel
    8: {'hog': True, 'raw': False, 'lbp': False, 'PCA': 0, 'kernelPCA': 'linear_kernel', 'strat': 'onevsone', 'SVM': 'CVXOPT', 'kernelSVM': 'gaussian_kernel', 'C': 1, 'sigma': 2, 'p': 1, 'subname': '1', 'gamma': 1, 'r': 1, 'bow': False, 'k': 1, 'sift': False, 'fishervect': False, 'with_norm': False}, 
    # Sigma on SVM only for hog for laplacian
    9: {'hog': True, 'raw': False, 'lbp': False, 'PCA': 0, 'kernelPCA': 'linear_kernel', 'strat': 'onevsone', 'SVM': 'CVXOPT', 'kernelSVM': 'laplacian_kernel', 'C': 1, 'sigma': 5, 'p': 1, 'subname': '1', 'gamma': 1, 'r': 1, 'bow': False, 'k': 1, 'sift': False, 'fishervect': False, 'with_norm': False}, 
    10: {'hog': True, 'raw': False, 'lbp': False, 'PCA': 0, 'kernelPCA': 'linear_kernel', 'strat': 'onevsone', 'SVM': 'CVXOPT', 'kernelSVM': 'laplacian_kernel', 'C': 1, 'sigma': 10, 'p': 1, 'subname': '1', 'gamma': 1, 'r': 1, 'bow': False, 'k': 1, 'sift': False, 'fishervect': False, 'with_norm': False}, 
    # Sigma on SVM only for hog for chi2
    11: {'hog': True, 'raw': False, 'lbp': False, 'PCA': 0, 'kernelPCA': 'linear_kernel', 'strat': 'onevsone', 'SVM': 'CVXOPT', 'kernelSVM': 'chi2_kernel', 'C': 1, 'sigma': 0.01, 'p': 1, 'subname': '1', 'gamma': 1, 'r': 1, 'bow': False, 'k': 1, 'sift': False, 'fishervect': False, 'with_norm': False}, 
    12: {'hog': True, 'raw': False, 'lbp': False, 'PCA': 0, 'kernelPCA': 'linear_kernel', 'strat': 'onevsone', 'SVM': 'CVXOPT', 'kernelSVM': 'chi2_kernel', 'C': 1, 'sigma': 0.1, 'p': 1, 'subname': '1', 'gamma': 1, 'r': 1, 'bow': False, 'k': 1, 'sift': False, 'fishervect': False, 'with_norm': False}, 
    # Bow with sift and linear SVM
    13: {'hog': False, 'raw': True, 'lbp': False, 'PCA': 0, 'kernelPCA': 'gaussian_kernel', 'strat': 'onevsone', 'SVM': 'CVXOPT', 'kernelSVM': 'linear_kernel', 'C': 5, 'sigma': 10, 'p': 1, 'subname': '1', 'gamma': 1, 'r': 1, 'bow': True, 'k': 10, 'sift': True, 'fishervect': False, 'with_norm': False}, 
    14: {'hog': False, 'raw': True, 'lbp': False, 'PCA': 0, 'kernelPCA': 'gaussian_kernel', 'strat': 'onevsone', 'SVM': 'CVXOPT', 'kernelSVM': 'linear_kernel', 'C': 5, 'sigma': 10, 'p': 1, 'subname': '1', 'gamma': 1, 'r': 1, 'bow': True, 'k': 20, 'sift': True, 'fishervect': False, 'with_norm': False}, 
     # Hog + raw RBF
    15: {'hog': True, 'raw': False, 'lbp': False, 'PCA': 0, 'kernelPCA': 'gaussian_kernel', 'strat': 'onevsone', 'SVM': 'CVXOPT', 'kernelSVM': 'gaussian_kernel', 'C': 5, 'sigma': 2, 'p': 1, 'subname': '1', 'gamma': 1, 'r': 1, 'bow': False, 'k': 50, 'sift': False, 'fishervect':False, 'with_norm': False}, 
    # Chi 2 + raw chi2 kernel
    16: {'hog': True, 'raw': False, 'lbp': False, 'PCA': 0, 'kernelPCA': 'gaussian_kernel', 'strat': 'onevsone', 'SVM': 'CVXOPT', 'kernelSVM': 'chi2_kernel', 'C': 5, 'sigma': 0.1, 'p': 1, 'subname': '1', 'gamma': 1, 'r': 1, 'bow': False, 'k': 50, 'sift': False, 'fishervect':False, 'with_norm': False}, 
    # SIFT + linear only
    17: {'hog': False, 'raw': True, 'lbp': False, 'PCA': 0, 'kernelPCA': 'gaussian_kernel', 'strat': 'onevsone', 'SVM': 'CVXOPT', 'kernelSVM': 'linear_kernel', 'C': 5, 'sigma': 10, 'p': 1, 'subname': '1', 'gamma': 1, 'r': 1, 'bow': False, 'k': 10, 'sift': True, 'fishervect': False, 'with_norm': False}, 
    
     # Hog + other features + linear SVM
    101: {'hog': True, 'raw': False, 'lbp': False, 'PCA': 0, 'kernelPCA': 'linear_kernel', 'strat': 'onevsone', 'SVM': 'CVXOPT', 'kernelSVM': 'linear_kernel', 'C': 1, 'sigma': 1, 'p': 1, 'subname': '1', 'gamma': 1, 'r': 1, 'bow': False, 'k': 1, 'sift': False, 'fishervect': False, 'with_norm': False, 'normalize':True}, 
    102: {'hog': True, 'raw': True, 'lbp': False, 'PCA': 0, 'kernelPCA': 'linear_kernel', 'strat': 'onevsone', 'SVM': 'CVXOPT', 'kernelSVM': 'linear_kernel', 'C': 1, 'sigma': 1, 'p': 1, 'subname': '1', 'gamma': 1, 'r': 1, 'bow': False, 'k': 1, 'sift': False, 'fishervect': False, 'with_norm': False, 'normalize':True},  
    103: {'hog': True, 'raw': False, 'lbp': True, 'PCA': 0, 'kernelPCA': 'linear_kernel', 'strat': 'onevsone', 'SVM': 'CVXOPT', 'kernelSVM': 'linear_kernel', 'C': 1, 'sigma': 1, 'p': 1, 'subname': '1', 'gamma': 1, 'r': 1, 'bow': False, 'k': 1, 'sift': False, 'fishervect': False, 'with_norm': False, 'normalize':True},
    104: {'hog': True, 'raw': False, 'lbp': False, 'PCA': 0, 'kernelPCA': 'linear_kernel', 'strat': 'onevsone', 'SVM': 'CVXOPT', 'kernelSVM': 'linear_kernel', 'C': 1, 'sigma': 1, 'p': 1, 'subname': '1', 'gamma': 1, 'r': 1, 'bow': False, 'k': 1, 'sift': True, 'fishervect': False, 'with_norm': False, 'normalize':True}, 
    105: {'hog': True, 'raw': True, 'lbp': True, 'PCA': 0, 'kernelPCA': 'linear_kernel', 'strat': 'onevsone', 'SVM': 'CVXOPT', 'kernelSVM': 'linear_kernel', 'C': 1, 'sigma': 1, 'p': 1, 'subname': '1', 'gamma': 1, 'r': 1, 'bow': False, 'k': 1, 'sift': True, 'fishervect': False, 'with_norm': False, 'normalize':True},
    # Bow with Sift + linear SVM + other features   
    106: {'hog': False, 'raw': True, 'lbp': False, 'PCA': 0, 'kernelPCA': 'linear_kernel', 'strat': 'onevsone', 'SVM': 'CVXOPT', 'kernelSVM': 'linear_kernel', 'C': 1, 'sigma': 1, 'p': 1, 'subname': '1', 'gamma': 1, 'r': 1, 'bow': True, 'k': 20, 'sift': True, 'fishervect': False, 'with_norm': False, 'normalize':True}, 
    107: {'hog': False, 'raw': False, 'lbp': True, 'PCA': 0, 'kernelPCA': 'linear_kernel', 'strat': 'onevsone', 'SVM': 'CVXOPT', 'kernelSVM': 'linear_kernel', 'C': 1, 'sigma': 1, 'p': 1, 'subname': '1', 'gamma': 1, 'r': 1, 'bow': True, 'k': 20, 'sift': True, 'fishervect': False, 'with_norm': False, 'normalize':True}, 
    # Hog gaussian kernel
    108: {'hog': True, 'raw': False, 'lbp': False, 'PCA': 0, 'kernelPCA': 'linear_kernel', 'strat': 'onevsone', 'SVM': 'CVXOPT', 'kernelSVM': 'gaussian_kernel', 'C': 1, 'sigma': 2, 'p': 1, 'subname': '1', 'gamma': 1, 'r': 1, 'bow': False, 'k': 1, 'sift': False, 'fishervect': False, 'with_norm': False, 'normalize':True}, 
    # Sigma on SVM only for hog for laplacian
    109: {'hog': True, 'raw': False, 'lbp': False, 'PCA': 0, 'kernelPCA': 'linear_kernel', 'strat': 'onevsone', 'SVM': 'CVXOPT', 'kernelSVM': 'laplacian_kernel', 'C': 1, 'sigma': 5, 'p': 1, 'subname': '1', 'gamma': 1, 'r': 1, 'bow': False, 'k': 1, 'sift': False, 'fishervect': False, 'with_norm': False, 'normalize':True}, 
    110: {'hog': True, 'raw': False, 'lbp': False, 'PCA': 0, 'kernelPCA': 'linear_kernel', 'strat': 'onevsone', 'SVM': 'CVXOPT', 'kernelSVM': 'laplacian_kernel', 'C': 1, 'sigma': 10, 'p': 1, 'subname': '1', 'gamma': 1, 'r': 1, 'bow': False, 'k': 1, 'sift': False, 'fishervect': False, 'with_norm': False, 'normalize':True}, 
    # Sigma on SVM only for hog for chi2
    111: {'hog': True, 'raw': False, 'lbp': False, 'PCA': 0, 'kernelPCA': 'linear_kernel', 'strat': 'onevsone', 'SVM': 'CVXOPT', 'kernelSVM': 'chi2_kernel', 'C': 1, 'sigma': 0.01, 'p': 1, 'subname': '1', 'gamma': 1, 'r': 1, 'bow': False, 'k': 1, 'sift': False, 'fishervect': False, 'with_norm': False, 'normalize':True}, 
    112: {'hog': True, 'raw': False, 'lbp': False, 'PCA': 0, 'kernelPCA': 'linear_kernel', 'strat': 'onevsone', 'SVM': 'CVXOPT', 'kernelSVM': 'chi2_kernel', 'C': 1, 'sigma': 0.1, 'p': 1, 'subname': '1', 'gamma': 1, 'r': 1, 'bow': False, 'k': 1, 'sift': False, 'fishervect': False, 'with_norm': False, 'normalize':True}, 
    # Bow with sift and linear SVM
    113: {'hog': False, 'raw': True, 'lbp': False, 'PCA': 0, 'kernelPCA': 'gaussian_kernel', 'strat': 'onevsone', 'SVM': 'CVXOPT', 'kernelSVM': 'linear_kernel', 'C': 5, 'sigma': 10, 'p': 1, 'subname': '1', 'gamma': 1, 'r': 1, 'bow': True, 'k': 10, 'sift': True, 'fishervect': False, 'with_norm': False, 'normalize':True}, 
    114: {'hog': False, 'raw': True, 'lbp': False, 'PCA': 0, 'kernelPCA': 'gaussian_kernel', 'strat': 'onevsone', 'SVM': 'CVXOPT', 'kernelSVM': 'linear_kernel', 'C': 5, 'sigma': 10, 'p': 1, 'subname': '1', 'gamma': 1, 'r': 1, 'bow': True, 'k': 20, 'sift': True, 'fishervect': False, 'with_norm': False, 'normalize':True}, 
     # Hog + raw RBF
    115: {'hog': True, 'raw': False, 'lbp': False, 'PCA': 0, 'kernelPCA': 'gaussian_kernel', 'strat': 'onevsone', 'SVM': 'CVXOPT', 'kernelSVM': 'gaussian_kernel', 'C': 5, 'sigma': 2, 'p': 1, 'subname': '1', 'gamma': 1, 'r': 1, 'bow': False, 'k': 50, 'sift': False, 'fishervect':False, 'with_norm': False, 'normalize':True}, 
    # Chi 2 + raw chi2 kernel
    116: {'hog': True, 'raw': False, 'lbp': False, 'PCA': 0, 'kernelPCA': 'gaussian_kernel', 'strat': 'onevsone', 'SVM': 'CVXOPT', 'kernelSVM': 'chi2_kernel', 'C': 5, 'sigma': 0.1, 'p': 1, 'subname': '1', 'gamma': 1, 'r': 1, 'bow': False, 'k': 50, 'sift': False, 'fishervect':False, 'with_norm': False, 'normalize':True}, 
    # SIFT + linear only
    117: {'hog': False, 'raw': True, 'lbp': False, 'PCA': 0, 'kernelPCA': 'gaussian_kernel', 'strat': 'onevsone', 'SVM': 'CVXOPT', 'kernelSVM': 'linear_kernel', 'C': 5, 'sigma': 10, 'p': 1, 'subname': '1', 'gamma': 1, 'r': 1, 'bow': False, 'k': 10, 'sift': True, 'fishervect': False, 'with_norm': False, 'normalize':True}, 
}

