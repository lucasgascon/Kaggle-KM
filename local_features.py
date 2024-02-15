import numpy as np
from sklearn.model_selection import train_test_split
import os
import time




# Function to calculate the Histogram of Oriented Gradients (HOG) for an image
def calculate_hog(image):
    
    
    # Convert the image to grayscale
    grayscale_image = np.dot(image, [0.2989, 0.5870, 0.1140])
    
    # Calculate the gradients in the x and y directions
    gradient_x = np.gradient(grayscale_image, axis=0)
    gradient_y = np.gradient(grayscale_image, axis=1)
    
    # Calculate the magnitude and direction of the gradients
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    direction = np.rad2deg(np.abs(np.arctan2(gradient_y, gradient_x)))
    
    # Define the number of bins for the histogram
    num_bins = 9
    step = 180 // num_bins
    cell_size = 8
    hist_bins = [20*i for i in range(9)]
    # Calculate the histogram of oriented gradients
    histogram_points_nine = np.zeros((32//cell_size, 32//cell_size, num_bins))

    # Window of 8x8 pixels
    for i in range(0, 32, 8): 
        temp = np.zeros((32//cell_size, num_bins))
        for j in range(0, 32, 8): 
            # Take magnitude and angle for the 8x8 window
            magnitude_values = magnitude[i:i+8, j:j+8]
            angle_values = direction[i:i+8, j:j+8]
            for k in range(cell_size): 
                for l in range(cell_size): 
                    
                    bins = np.zeros(num_bins)
                
                    # Calculate the id of the bin 
                    id_bin_j = int(angle_values[k,l]//step)
                    
                    if id_bin_j == 9:
                        id_bin_j = 8
                    
                    # Calculate the value to put in te bin
                    Vj = magnitude_values[k,l] * np.abs(hist_bins[id_bin_j] - angle_values[k,l]) / step
 
                    
                    if id_bin_j == 8: 
                        bins[id_bin_j] += Vj 
                    else:
                        Vjplus1 = magnitude_values[k,l] * np.abs(hist_bins[id_bin_j+1] - angle_values[k,l]) / step
                        bins[id_bin_j] += Vj 
                        bins[id_bin_j+1] += Vjplus1

            temp[j//cell_size,:] = bins
        # print('Length of temp',temp.shape)
        histogram_points_nine[i//cell_size,:,:] = temp
    # print('Global length',histogram_points_nine.shape)
                    
    # Pooling in 4 blocks and Normalization 
    feature_vectors = np.zeros((32//cell_size - 1,32//cell_size - 1, num_bins*4))
    for i in range(0, histogram_points_nine.shape[0] - 1, 1):   
        for j in range(0,histogram_points_nine.shape[1] - 1, 1): 
            values = histogram_points_nine[i:i+2,j:j+2,:]
            k = np.sqrt(np.sum(values**2))  
            feature_vectors[i,j,:] = values.flatten() / (k + 1e-8)
    

    # Return the feature vectors
    # print(feature_vectors.shape)
    return np.array(feature_vectors).flatten()