import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils import row_to_image, viz_image
import os
from SVMs import BinarySVM, SVM_SGD
from kernels import linear_kernel, polynomial_kernel, gaussian_kernel, sigmoid_kernel, laplacian_kernel, chi2_kernel
from local_features import calculate_hog, calculate_LocalBinaryPattern, calculate_SIFT
from global_features import kernelPCA, BoW, FisherVectorGMM
from tqdm import tqdm
import argparse

params = {'C': 1.0,
        'sigma': 5.0,
        'p': 3,
        'learning_rate': 0.01,
        'epochs': 1000,
        'gamma': 1.0,
        'r': 0.0,
        }
kernels = {'linear_kernel': linear_kernel,
           'polynomial_kernel': polynomial_kernel,
           'gaussian_kernel': gaussian_kernel,
           'sigmoid_kernel': sigmoid_kernel,
           'laplacian_kernel': laplacian_kernel}
    
def main(args):
    
    # Set up parameters for the SVM
    params['C'] = args.C
    params['sigma'] = args.sigma
    params['p'] = args.p
    params['learning_rate'] = args.learning_rate
    params['epochs'] = args.epochs
    params['gamma'] = args.gamma
    params['r'] = args.r
    
    if args.subname == '0':
        if args.kernelSVM == 'gaussian_kernel' or args.kernelPCA == 'gaussian_kernel' or args.kernelSVM == 'laplacian_kernel' or args.kernelPCA == 'laplacian_kernel':
            if args.bow or args.fishervect:
                args.subname = args.kernelSVM + '_'+args.kernelPCA+ '_C' + str(args.C) + '_sigma' + str(args.sigma) + '_PCA' + str(args.PCA) + '_bow' + str(args.bow) + '_k' + str(args.k) +  '_fishervect' + str(args.fishervect)
            else:
                args.subname = args.kernelSVM + '_'+args.kernelPCA+ '_C' + str(args.C) + '_sigma' + str(args.sigma) + '_PCA' + str(args.PCA) + '_hog' + str(args.hog) + '_raw' + str(args.raw) + '_lbp' + str(args.lbp)
        else:
            if args.bow or args.fishervect:
                args.subname = args.kernelSVM + '_'+args.kernelPCA+ '_C' + str(args.C) + '_PCA' + str(args.PCA) + '_bow' + str(args.bow) + '_k' + str(args.k) + '_fishervect' + str(args.fishervect)
            else:
                args.subname = args.kernelSVM + '_'+args.kernelPCA+ '_C' + str(args.C) + '_PCA' + str(args.PCA) + '_hog' + str(args.hog) + '_raw' + str(args.raw) + '_lbp' + str(args.lbp)
    
    """Create feature vectors for the training and test sets.
    """
    
    # Load the dataset
    trainset = pd.read_csv('data/Xtr.csv', header=None)  # Assuming there's no header row
    trainset = trainset.iloc[:,:-1]
    train_labels = pd.read_csv('data/Ytr.csv').iloc[:, 1].values

    # For testset
    testset = pd.read_csv('data/Xte.csv', header=None)
    testset = testset.iloc[:,:-1]
        
    # For trainset
    train_images = np.array([row_to_image(row.values) for index, row in trainset.iterrows()])
    # train_images = (train_images - np.mean(train_images, axis=0))/np.std(train_images, axis=0)
    
    
    # For testset
    test_images = np.array([row_to_image(row.values)  for index, row in testset.iterrows()])
    # test_images = (test_images - np.mean(test_images, axis=0))/np.std(test_images, axis=0)
    
    train_vector = None
    test_vector = None
    
    # Set kernel parameters:
    if args.kernelSVM == 'polynomial_kernel':
        kernels[args.kernelSVM] = lambda x,y: polynomial_kernel(x,y, params['p' ])
    elif args.kernelSVM == 'gaussian_kernel':
        kernels[args.kernelSVM] = lambda x,y: gaussian_kernel(x,y, params['sigma'])
    elif args.kernelSVM == 'sigmoid_kernel':
        kernels[args.kernelSVM] = lambda x,y: sigmoid_kernel(x,y, params['gamma'], params['r'])
    elif args.kernelSVM == 'laplacian_kernel':
        kernels[args.kernelSVM] = lambda x,y: laplacian_kernel(x,y, params['sigma'])
    elif args.kernelSVM == 'chi2_kernel':
        kernels[args.kernelSVM] = lambda x,y: chi2_kernel(x,y, params['sigma'])
    
    if args.kernelPCA == 'polynomial_kernel':
        kernels[args.kernelPCA] = lambda x,y: polynomial_kernel(x,y, params['p' ])
    elif args.kernelPCA == 'gaussian_kernel':
        kernels[args.kernelPCA] = lambda x,y: gaussian_kernel(x,y, params['sigma'])
    elif args.kernelPCA == 'sigmoid_kernel':
        kernels[args.kernelPCA] = lambda x,y: sigmoid_kernel(x,y, params['gamma'], params['r'])
    elif args.kernelPCA == 'laplacian_kernel':
        kernels[args.kernelPCA] = lambda x,y: laplacian_kernel(x,y, params['sigma'])
    elif args.kernelPCA == 'chi2_kernel':
        kernels[args.kernelPCA] = lambda x,y: chi2_kernel(x,y, params['sigma'])
        
    if args.bow:
        if args.hog:
            print("Calculating HOG features")
            list_train_features = [calculate_hog(image,local_descriptor=True) for image in tqdm(train_images)]
            list_test_features = [calculate_hog(image,local_descriptor=True) for image in tqdm(test_images)]
            args.hog = False
        elif args.sift:
            print("Calculating SIFT features")
            list_train_features = [calculate_SIFT(image) for image in tqdm(train_images)]
            list_test_features = [calculate_SIFT(image) for image in tqdm(test_images)]
            args.sift = False
        
        print("Calculating Bag of Words features")
        bow = BoW(args.k)
        train_features = bow.fit(list_train_features)
        test_features = bow.predict(list_test_features)

        if train_vector is None:
            train_vector = train_features
            test_vector = test_features
        else:
            train_vector = np.concatenate((train_vector, train_features), axis=1)
            test_vector = np.concatenate((test_vector, test_features), axis=1)
            
    if args.fishervect:
        if args.hog:
            print("Calculating HOG features")
            list_train_features = [calculate_hog(image,local_descriptor=True) for image in tqdm(train_images)]
            list_test_features = [calculate_hog(image,local_descriptor=True) for image in tqdm(test_images)]
            args.hog = False
        elif args.sift:
            print("Calculating SIFT features")
            list_train_features = [calculate_SIFT(image) for image in tqdm(train_images)]
            list_test_features = [calculate_SIFT(image) for image in tqdm(test_images)]
            args.sift = False
        
        print("Calculating Fisher Vector features")
        fishvect = FisherVectorGMM(args.k)
        fishvect.fit(list_train_features)
        train_features = np.array([fishvect.transform(trainfeatures) for trainfeatures in tqdm(list_train_features)])
        test_features = np.array([fishvect.transform(testfeatures) for testfeatures in tqdm(list_test_features)])

        if train_vector is None:
            train_vector = train_features
            test_vector = test_features
        else:
            train_vector = np.concatenate((train_vector, train_features), axis=1)
            test_vector = np.concatenate((test_vector, test_features), axis=1)
            
    if args.hog:
        # Calculate the HOG for each image
        print("Calculating HOG features")
        train_hog_features = np.array([calculate_hog(image) for image in tqdm(train_images)])
        test_hog_features = np.array([calculate_hog(image) for image in tqdm(test_images)])
        
        if train_vector is None:
            train_vector = train_hog_features
            test_vector = test_hog_features
        else:
            train_vector = np.concatenate((train_vector, train_hog_features), axis=1)
            test_vector = np.concatenate((test_vector, test_hog_features), axis=1)
        
    if args.raw: 
        if train_vector is None:
            train_vector = train_images.flatten().reshape(-1, 3*32*32)
            test_vector = test_images.flatten().reshape(-1, 3*32*32)
            
        else:
            train_vector = np.concatenate((train_vector, train_images.reshape(-1, 3*32*32)), axis=1)
            test_vector = np.concatenate((test_vector, test_images.reshape(-1, 3*32*32)), axis=1)
    
    if args.lbp: 
        print("Calculating LocalBinaryPattern features")
        train_lbs_features = np.array([calculate_LocalBinaryPattern(image) for image in tqdm(train_images)])
        test_lbs_features = np.array([calculate_LocalBinaryPattern(image) for image in tqdm(test_images)])
        
        if train_vector is None:
            train_vector = train_lbs_features
            test_vector = test_lbs_features
        else:
            train_vector = np.concatenate((train_vector, train_lbs_features), axis=1)
            test_vector = np.concatenate((test_vector, test_lbs_features), axis=1)
            
    if args.sift:
        print("Calculating SIFT features")
        train_sift_features = np.array([np.mean(calculate_SIFT(image),axis=0)for image in tqdm(train_images)])
        test_sift_features = np.array([np.mean(calculate_SIFT(image),axis=0) for image in tqdm(test_images)])
        
        if train_vector is None:
            train_vector = train_sift_features
            test_vector = test_sift_features
        else:
            train_vector = np.concatenate((train_vector, train_sift_features), axis=1)
            test_vector = np.concatenate((test_vector, test_sift_features), axis=1)
    
    # Split the data into training and validation sets
    if not args.tosubmit:
        X_train, X_val, y_train, y_val = train_test_split(train_vector, train_labels, test_size=0.1, random_state = 29)
    else:
        X_train = train_vector
        y_train = train_labels

    if args.PCA != 0:
        kernel_pca = kernelPCA(kernels[args.kernelPCA], args.PCA, with_norm = args.with_norm) 
        
        # train_vector = train_vector - np.mean(train_vector, axis=0)
        # test_vector = test_vector - np.mean(test_vector, axis=0)
        X_train = kernel_pca.fit(X_train)
        
        # If 80% enables to reach the same number of components
        if not args.to_submit:
            X_val = kernel_pca.transform(X_val)
            
        test_vector = kernel_pca.transform(test_vector)
        

    print('KernelSVM:', args.kernelSVM,'    KernelPCA:', args.kernelPCA, )
    print('Features HOG:', args.hog,'   Features raw:', args.raw, ' Features LBP:', args.lbp, ' Features BOW:',
          args.bow, ' Features SIFT:', args.sift, ' Features FisherVector:', args.fishervect)
    print('Dimension PCA:', train_vector.shape[1])
    print('C:', args.C, '   Sigma:', args.sigma)
    
    """Train a multi-class classifier using the One-vs-All approach.
    """
    if args.strat == 'onevsall':
    # Create a dictionary to store the trained models
        models = {}
        print('Training OneVsAll model')
        for class_label in np.unique(train_labels): 
            print('Training model for class', class_label)
            # Create a binary label vector for the current class
            binary_labels = np.where(y_train == class_label, 1, -1)   
            # Create a SVM model
            if args.SVM == 'SGD':
                model = SVM_SGD(C = params['C'], kernel= kernels[args.kernelSVM],
                                epochs=params['epochs'], learning_rate= params['learning_rate'])               
            elif args.SVM == 'CVXOPT':
                model = BinarySVM(kernel = kernels[args.kernelSVM], C = params['C'])
            # Fit the model on the training data
            model.fit(X_train, binary_labels)
            # Store the trained model in the dictionary
            models[class_label] = model

        # Predict the class labels for the validation set
        if not args.tosubmit:
            predictions = []
            for features in X_val:  
                if args.SVM == 'SGD':
                    class_scores = [model.predict(features) for model in models.values()]
                    
                elif args.SVM == 'CVXOPT':
                    class_scores = [model._decision_function([features])[0] for model in models.values()]
                # Select the class label with the highest score 
                # For each label, class score is applied and then take the label which obtains the max class scores
                predicted_class = max(models.keys(), key=lambda x: class_scores[x])
                
                # Append the predicted class label to the list of predictions
                predictions.append(predicted_class)
            # Calculate the accuracy of the OneVSAll model
            accuracy = np.mean(predictions == y_val)
            print("Accuracy: {:.2f}%".format(accuracy * 100))


        test_predictions = []
        print('Predicting test set')
        for features in tqdm(test_vector):
            if args.SVM == 'SGD':
                class_scores = [model.predict(features) for model in models.values()]
                
            elif args.SVM == 'CVXOPT':
                class_scores = [model._decision_function([features])[0] for model in models.values()]
        
            predicted_class = max(models.keys(), key=lambda x: class_scores[x])
            test_predictions.append(predicted_class)
        test_predictions_df = pd.DataFrame({'Prediction' : test_predictions})
        test_predictions_df.index += 1
        test_predictions_df

        # Save the predictions
        submission_filename = "submission_"+args.subname+".csv"
        submission_filepath = os.path.join("submissions", submission_filename)
        test_predictions_df.to_csv(submission_filepath, index_label='Id')

        # Print the path to the submission file
        print("Submission saved at:", submission_filepath)
    
        """Train a multi-class classifier using the One-vs-All approach.
        """    
    elif args.strat == 'onevsone':
        models = {}
        print('Training OneVsOne model')
        for id, first_class_label in enumerate(np.sort(np.unique(train_labels))[:-1]):
            for second_class_label in np.sort(np.unique(train_labels))[id+1:]:
                print('Training model for classes', first_class_label, 'and', second_class_label)
                
                # Selection the data
                X_train_partial = X_train[np.where((y_train == first_class_label) | (y_train == second_class_label))]
                y_train_partial = y_train[np.where((y_train == first_class_label) | (y_train == second_class_label))]
                               
                # Create a binary label vector for the current class and the other class
                binary_labels = np.where(y_train_partial == first_class_label, 1, -1)
                           
                if args.SVM == 'SGD':
                    model = SVM_SGD(C = params['C'], kernel= kernels[args.kernelSVM],
                                    epochs=params['epochs'], learning_rate= params['learning_rate'])                  
                elif args.SVM == 'CVXOPT':
                    model = BinarySVM(kernel = kernels[args.kernelSVM], C = params['C'])           
                # Fit the model on the training data
                model.fit(X_train_partial, binary_labels)    
                models[str(first_class_label)+"_"+str(second_class_label)] = model
        if not args.tosubmit:              
            predictions = []
            for features in X_val:
                labels = {key: 0 for key in np.sort(np.unique(train_labels))}
                for id, first_class_label in enumerate(np.sort(np.unique(train_labels))[:-1]):
                    for second_class_label in np.sort(np.unique(train_labels))[id+1:]:
                        
                        if args.SVM == 'SGD':
                            pred = models[str(first_class_label)+"_"+str(second_class_label)].predict(features)
                        elif args.SVM == 'CVXOPT':
                            pred = models[str(first_class_label)+"_"+str(second_class_label)]._decision_function([features])[0]

                        if pred == 1:
                            labels[first_class_label] += 1
                        else:   
                            labels[second_class_label] += 1
                        
                # Select the class label with the highest score
                predicted_class = max(labels.keys(), key=lambda x: labels[x])
                predictions.append(predicted_class)
            # Calculate the accuracy of the OneVSOne model
            accuracy = np.mean(predictions == y_val)
            print("Accuracy: {:.2f}%".format(accuracy * 100))

        test_predictions = []
        
        print('Predicting test set')
        for features in tqdm(test_vector):
            labels = {key: 0 for key in np.sort(np.unique(train_labels))}
            for id, first_class_label in enumerate(np.sort(np.unique(train_labels))[:-1]):
                for second_class_label in np.sort(np.unique(train_labels))[id+1:]:
                    
                    if args.SVM == 'SGD':
                         pred = models[str(first_class_label)+"_"+str(second_class_label)].predict(features)
                                         
                    elif args.SVM == 'CVXOPT':
                         pred = models[str(first_class_label)+"_"+str(second_class_label)]._decision_function([features])[0]

                    if pred == 1:
                        labels[first_class_label] += 1
                    else:   
                        labels[second_class_label] += 1
            # If several labels are predicted, take a random one
            key_list = np.random.permutation(list(labels.keys()))
            predicted_class = max(key_list, key=lambda x: labels[x])
            test_predictions.append(predicted_class)

        test_predictions_df = pd.DataFrame({'Prediction' : test_predictions})
        test_predictions_df.index += 1

        submission_filename = "submission_"+args.subname+".csv"
        submission_filepath = os.path.join("submissions", submission_filename)
        test_predictions_df.to_csv(submission_filepath, index_label='Id')
        print("Submission saved at:", submission_filepath)
    if not args.tosubmit:
        return accuracy
    else:
        return None
 
    
"""Parser arguments
"""
def parser_args(parser):

    parser.add_argument('--hog', action = 'store_true', help='Use HOG features')
    parser.add_argument('--raw', action = 'store_true', help='Use raw pixel features')
    parser.add_argument('--lbp', action = 'store_true', help='Use Local Binary Pattern features')
    parser.add_argument('--PCA', type = int, default = 0, help='Number of components of PCA, 0 if not used and -1 if above 80 perc of variance is used')
    parser.add_argument('--kernelPCA', type = str, default = 'linear_kernel', help='Kernel for PCA')
    parser.add_argument('--strat', type = str, default = 'onevsone', help='Use One vs One or One vs All approach')
    parser.add_argument('--SVM', type = str, default = 'SGD', help='SVM algorithm to use')
    parser.add_argument('--kernelSVM', type = str, default = 'linear_kernel', help='Kernel for SVM')
    parser.add_argument('--C', type = float, default = 1.0, help='Regularization parameter')
    parser.add_argument('--sigma', type = float, default = 1.0, help='Sigma for gaussian kernel')
    parser.add_argument('--p', type = int, default = 3, help='Degree for polynomial kernel')
    parser.add_argument('--learning_rate', type = float, default = 0.01, help='Learning rate for SGD')
    parser.add_argument('--epochs', type = int, default = 1000, help='Number of epochs for SGD')
    parser.add_argument('--subname', type = str, default = '0', help='Name of the submission file')
    parser.add_argument('--gamma', type = float, default = 1., help='Gamma for sigmoid kernel')
    parser.add_argument('--r', type = float, default = 1., help='r for sigmoid kernel')
    parser.add_argument('--tosubmit', action = 'store_true', help='Choose whether to train on the whole dataset or not')
    parser.add_argument('--bow', action = 'store_true', help='Use Bag of Words features')
    parser.add_argument('--k', type = int, default = 10, help='Number of clusters for Bag of Words or of GMM components for Fisher Vector')
    parser.add_argument('--sift', action = 'store_true' , help='Use SIFT features')
    parser.add_argument('--fishervect', action = 'store_true', help='Use fisher vector features')
    parser.add_argument('--with_norm', action = 'store_true', help='Normalize gram matrix in PCA')

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = parser_args(parser) 
    args = parser.parse_args()

    main(args)
    
    
