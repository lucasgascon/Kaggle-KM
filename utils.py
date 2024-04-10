import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


"""Create RGB image from csv row
"""


def row_to_image(row, normalize=True):
    # Split the row into three parts, for R, G, and B channels
    if normalize:
        R = (row[:1024].reshape(32, 32) - np.min(row[:1024])) / \
            (np.max(row[:1024]) - np.min(row[:1024]))
        G = (row[1024:2048].reshape(32, 32) - np.min(row[1024:2048])
             ) / (np.max(row[1024:2048]) - np.min(row[1024:2048]))
        B = (row[2048:].reshape(32, 32) - np.min(row[2048:])) / \
            (np.max(row[2048:]) - np.min(row[2048:]))
        # Stack the channels along the last dimension to form an RGB image
    else:
        R = row[:1024].reshape(32, 32)
        G = row[1024:2048].reshape(32, 32)
        B = row[2048:].reshape(32, 32)
    image = np.stack([R, G, B], axis=-1)
    return image


"""Vizualize the image
"""


def viz_image(image, ax=None, exp_norm=True):
    new_im = np.zeros(image.shape)
    for i in range(3):
        if exp_norm:
            new_im[:, :, i] = 1 / (1 + np.exp(-10*image[:, :, i]))
        else:
            new_im[:, :, i] = (image[:, :, i] - np.min(image[:, :, i])) / \
                (np.max(image[:, :, i]) - np.min(image[:, :, i]))
    if ax is None:
        fig, ax = plt.subplots()
    ax.imshow(new_im)
    ax.axis('off')
    return ax


"""Average the results of several models from submissions placed in a directory
"""


def average_submissions(dir_path="submissions/averaging", new_name="submissions/averaged_submission.csv"):
    for i, sub_files in enumerate(os.listdir(dir_path)):
        sub = pd.read_csv(os.path.join(dir_path, sub_files))

        if i == 0:
            dict_averaged_label = {row[1]['Id']: [
                row[1]['Prediction']] for row in sub.iterrows()}
        else:
            for row in sub.iterrows():
                dict_averaged_label[row[1]['Id']].append(row[1]['Prediction'])

    test_predictions = []
    count_labels = {k: 0 for k in range(10)}
    for k, v in dict_averaged_label.items():
        labels, counts = np.unique(v, return_counts=True)
        id_max_count = np.argmax(counts)
        ids_max_count = np.where(counts == counts[id_max_count])[0]
        if len(ids_max_count) > 1:
            p = np.zeros(len(ids_max_count))
            for i, lab in enumerate(labels[ids_max_count]):
                p[i] = 1/(count_labels[lab] + 1)
            p = p/p.sum()
            label = np.random.choice(labels[ids_max_count], 1, p=p)[0]
            count_labels[label] += 1
            test_predictions.append(label)
        else:
            label = labels[ids_max_count][0]
            count_labels[label] += 1
            test_predictions.append(label)
    test_predictions_df = pd.DataFrame(
        {'Prediction': test_predictions}, index=dict_averaged_label.keys())
    test_predictions_df.to_csv(new_name, index_label='Id')


if __name__ == "__main__":
    # Example usage
    average_submissions()
    print("Averaging done!")
