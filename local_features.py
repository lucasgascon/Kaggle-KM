import numpy as np

# Function to calculate the Histogram of Oriented Gradients (HOG) for an image
def calculate_hog(image, local_descriptor = False):
    
    
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
    if local_descriptor: 
        return np.array(feature_vectors).reshape((32//cell_size - 1)**2,num_bins*4)
    else:
        return np.array(feature_vectors).flatten()





def get_neighbour(image, center, x, y):
    value = 0  
    try: 
        if image[x,y] >= center: 
            value = 1
    except: 
        pass
    return value
    
def calculate_LocalBinaryPattern(image):
    # Convert the image to grayscale
    grayscale_image = np.dot(image, [0.2989, 0.5870, 0.1140])
    
    height, width = grayscale_image.shape
    img_lbp = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            neigh_val = np.zeros(8)
            center = image[i,j]
            
            # Clockwise order
            neigh_val[0] = get_neighbour(image, center, i-1, j-1)
            neigh_val[1] = get_neighbour(image, center, i-1, j)
            neigh_val[2] = get_neighbour(image, center, i-1, j+1)
            neigh_val[3] = get_neighbour(image, center, i, j+1)
            neigh_val[4] = get_neighbour(image, center, i+1, j+1)
            neigh_val[5] = get_neighbour(image, center, i+1, j)
            neigh_val[6] = get_neighbour(image, center, i+1, j-1)
            neigh_val[7] = get_neighbour(image, center, i, j-1)
            
            power_base = [1, 2, 4, 8, 16, 32, 64, 128] 
            img_lbp[i,j] = np.sum(neigh_val * power_base)
            
    return img_lbp.flatten()
"""
# À Vérifier !!!!!
def SIFT(image):
    # Convert the image to grayscale
    grayscale_image = np.dot(image, [0.2989, 0.5870, 0.1140])
    
    # Define the scale space parameters
    num_octaves = 4
    num_scales = 5
    sigma = 1.6
    
    # Create the scale space
    scale_space = np.zeros((num_octaves, num_scales, grayscale_image.shape[0], grayscale_image.shape[1]))
    octave_images = []
    
    # Generate the octave images
    for octave in range(num_octaves):
        octave_images.append(grayscale_image)
        for scale in range(1, num_scales):
            scaled_image = cv2.resize(octave_images[octave], None, fx=1/(2**scale), fy=1/(2**scale))
            scaled_image = cv2.GaussianBlur(scaled_image, (0, 0), sigmaX=sigma*(2**scale), sigmaY=sigma*(2**scale))
            octave_images.append(scaled_image)
    
    # Calculate the Difference of Gaussians (DoG)
    dog = np.zeros((num_octaves, num_scales-1, grayscale_image.shape[0], grayscale_image.shape[1]))
    for octave in range(num_octaves):
        for scale in range(num_scales-1):
            dog[octave, scale] = octave_images[octave][:, :] - octave_images[octave][scale+1][:, :]
    
    # Find the keypoints
    keypoints = []
    for octave in range(num_octaves):
        for scale in range(1, num_scales-2):
            for i in range(1, dog[octave, scale].shape[0]-1):
                for j in range(1, dog[octave, scale].shape[1]-1):
                    if is_extremum(dog, octave, scale, i, j):
                        keypoints.append((i, j))
    
    # Compute the descriptors
    descriptors = []
    for keypoint in keypoints:
        descriptor = compute_descriptor(grayscale_image, keypoint)
        descriptors.append(descriptor)
    
    return np.array(descriptors)


def is_extremum(dog, octave, scale, i, j):
    value = dog[octave, scale, i, j]
    if value > 0:
        for x in range(-1, 2):
            for y in range(-1, 2):
                if dog[octave, scale-1, i+x, j+y] >= value or dog[octave, scale+1, i+x, j+y] >= value:
                    return False
    else:
        for x in range(-1, 2):
            for y in range(-1, 2):
                if dog[octave, scale-1, i+x, j+y] <= value or dog[octave, scale+1, i+x, j+y] <= value:
                    return False
    return True


def compute_descriptor(image, keypoint):
    descriptor = []
    x, y = keypoint
    patch = image[x-8:x+8, y-8:y+8]
    patch = cv2.resize(patch, (16, 16))
    patch = patch.flatten()
    descriptor.append(patch)
    return np.array(descriptor)
    
""" 
            
    