import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils import row_to_image, viz_image
import os
import time
from SVMs import BinarySVM, SVM_SGD
from kernels import linear_kernel, polynomial_kernel, gaussian_kernel, sigmoid_kernel, laplacian_kernel
from local_features import calculate_hog, calculate_LocalBinaryPattern
from global_features import compute_kernelPCA
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
        args.subname = 'strat_'+args.strat+'_kernelSVM_'+args.kernelSVM+'_PCA_'+str(args.PCA)+'_hog_'+str(args.hog)+'_raw_'+str(args.raw)+'_kernelPCA_'+args.kernelPCA+'_C_'+str(args.C)+'_sigma_'+str(args.sigma)+'_p_'+str(args.p)
    
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
    
    # For testset
    test_images = np.array([row_to_image(row.values)  for index, row in testset.iterrows()])
    

    train_vector = None
    test_vector = None
    
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
        
    
    if args.PCA != 0: 
        train_vector = compute_kernelPCA(train_vector, kernels[args.kernelPCA], args.PCA)
        # If 80% enables to reach the same number of components
        test_vector = compute_kernelPCA(test_vector, kernels[args.kernelPCA], train_vector.shape[1])
    print('Train vector shape:', train_vector.shape)    
    print('Test vector shape:', test_vector.shape)
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(train_vector, train_labels, test_size=0.1, random_state=42)
    
    print('Kernel:', args.kernelSVM)
    print('Features HOG:', args.hog)
    print('Features raw:', args.raw)
    print('PCA:', args.PCA)
    
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
                        
            predicted_class = max(labels.keys(), key=lambda x: labels[x])
            test_predictions.append(predicted_class)

        test_predictions_df = pd.DataFrame({'Prediction' : test_predictions})
        test_predictions_df.index += 1

        submission_filename = "submission_"+args.subname+".csv"
        submission_filepath = os.path.join("submissions", submission_filename)
        test_predictions_df.to_csv(submission_filepath, index_label='Id')
        print("Submission saved at:", submission_filepath)
 
    
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

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = parser_args(parser) 
    args = parser.parse_args()

    main(args)
    
    
