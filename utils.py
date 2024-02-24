import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# Function to convert a row to an image
def row_to_image(row):
    # Split the row into three parts, for R, G, and B channels
    R = (row[:1024].reshape(32, 32) - np.min(row[:1024])) / (np.max(row[:1024]) - np.min(row[:1024]))
    G = (row[1024:2048].reshape(32, 32) - np.min(row[1024:2048])) / (np.max(row[1024:2048]) - np.min(row[1024:2048]))
    B = (row[2048:].reshape(32, 32) - np.min(row[2048:])) / (np.max(row[2048:]) - np.min(row[2048:]))
    # Stack the channels along the last dimension to form an RGB image
    image = np.stack([R, G, B], axis=-1)
    return image

def viz_image(image, ax=None, exp_norm = True):
    new_im = np.zeros(image.shape)
    for i in range(3):
        if exp_norm:
            new_im[:,:,i] = 1/ (1 + np.exp(-10*image[:,:,i]))
        else:
            new_im[:,:,i] = (image[:,:,i] - np.min(image[:,:,i]) )/ (np.max(image[:,:,i]) -  np.min(image[:,:,i]))
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(new_im)
    return ax

# Average the results of several models from submissions placed in a directory
def average_submissions(dir_path = "submissions/averaging", new_name = "averaged_submission.csv"):
    for i, sub_files in enumerate(os.listdir(dir_path)):
        sub = pd.read_csv(os.path.join(dir_path, sub_files))
        
        if i == 0:    
            dict_averaged_label = {row[1]['Id']: [row[1]['Prediction']] for row in sub.iterrows()}
        else:
            for row in sub.iterrows():
                dict_averaged_label[row[1]['Id']].append(row[1]['Prediction'])
    
    test_predictions = []
    for k, v in dict_averaged_label.items():
        labels, counts = np.unique(v, return_counts=True)
        id_max_count = np.argmax(counts)
        ids_max_count = np.where(counts == counts[id_max_count])[0]
        test_predictions.append(np.random.choice(labels[ids_max_count],1)[0])
    test_predictions_df = pd.DataFrame({'Prediction' : test_predictions},index = dict_averaged_label.keys())
    test_predictions_df.to_csv('submissions/'+new_name, index_label='Id')
    
if __name__ == "__main__":
    # Example usage
    average_submissions()
    print("Averaging done!")

