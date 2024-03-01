import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from skimage.feature import hog, ORB
import cv2

# Function to calculate the Histogram of Oriented Gradients (HOG) for an image
def calculate_hog(image, local_descriptor = False, handmade = False):
    

    if handmade:
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
    else:
        # Calculate the HOG using the skimage library
        if local_descriptor: 
            fd = hog(image, orientations=9, pixels_per_cell=(8, 8),
                                cells_per_block=(2, 2), channel_axis=-1, feature_vector=False)
            fd = fd.reshape(9,9*2*2)
            return fd
        else:
            feature_vectors = hog(grayscale_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), channel_axis=-1,feature_vector=True, orientations=9)
            return feature_vectors
        
def calculate_ORB(image, n_keypoints = 200):
        # Convert the image to grayscale
        grayscale_image = np.dot(image, [0.2989, 0.5870, 0.1140])
        descriptor_extractor = ORB(n_keypoints=n_keypoints)

        descriptor_extractor.detect_and_extract(grayscale_image)
        keypoints = descriptor_extractor.keypoints
        descriptors = descriptor_extractor.descriptors
        return descriptors
        




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


def calculate_SIFT(image, sigma = 1.6, num_octaves = 4, num_scales = 5, handmade = False):
    # Convert the image to grayscale
    grayscale_image = np.dot(image, [0.2989, 0.5870, 0.1140])
    
    if handmade:
        # Create the scale space
        octave_images = []
        
        """Generate base image from input image by upsampling by 2 in both directions and blurring
        """
        # Upsampling by a factor of 2
        image = zoom(grayscale_image, (2, 2, 1), order=1)

        k = 2**(1/num_scales)
        # Generate the octave images
        pyr_images = []
        for octave in range(num_octaves):
            new_sigma = sigma
            scaled_image = zoom(image, (1/2**(octave+1),1/2**(octave+1),1), order=1)
            octave_images.append(scaled_image)
            for scale in range(1, num_scales):
                blurred_image = gaussian_filter(scaled_image, sigma=(sigma*(2**scale), sigma*(2**scale)))
                pyr_images.append(blurred_image)
                new_sigma = new_sigma * k
        
        # Calculate the Difference of Gaussians (DoG)
        dog = []
        for octave in range(num_octaves):
            octave_dogs = np.zeros((num_scales-1, pyr_images[octave*num_scales].shape[0], pyr_images[octave*num_scales].shape[1]))
            for scale in range(num_scales-1):
                octave_dogs[scale,:,:] = pyr_images[octave+scale]- pyr_images[octave+scale+1]
            dog.append(octave_dogs)
        
        
        # Find the keypoints
        keypoints = []
        for octave in range(num_octaves):
            for scale in range(1, num_scales-2):
                for i in range(1, dog[octave+scale].shape[0]-1):
                    for j in range(1, dog[octave+scale].shape[1]-1):
                        if is_extremum(dog, octave, scale, i, j):
                            keypoints.append((i, j))
        
        # Compute the descriptors
        descriptors = []
        for keypoint in keypoints:
            descriptor = compute_descriptor(grayscale_image, keypoint)
            descriptors.append(descriptor)
        return np.array(descriptors)
    else:
        rescaled_image = (image*255).astype(dtype=np.uint8)
        cv2_image = cv2.cvtColor(rescaled_image, cv2.COLOR_RGB2GRAY) 
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(cv2_image, None)
        if descriptors is None:
            return np.zeros((1, 128))
        return descriptors


def is_extremum(dog, octave, scale, i, j):
    value = dog[octave+scale][i, j]
    if value > 0:
        for x in range(-1, 2):
            for y in range(-1, 2):
                if dog[octave+scale-1][i+x, j+y] >= value or dog[octave+scale+1][i+x, j+y] >= value:
                    return False
    else:
        for x in range(-1, 2):
            for y in range(-1, 2):
                if dog[octave+scale-1][i+x, j+y]<= value or dog[octave+scale+1][i+x, j+y] <= value:
                    return False
    return True

# À revoir
def compute_descriptor(image, keypoint):
    descriptor = []
    x, y = keypoint
    patch = image[x-8:x+8, y-8:y+8]
    patch = patch.reshape(16, 16)
    patch = patch.flatten()
    descriptor.append(patch)
    return np.array(descriptor)


    

            
    