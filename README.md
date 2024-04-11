# Kaggle Challenge: Image Classification with Machine Learning Kernel Methods

## Introduction

Welcome to our Kaggle challenge for image classification with 10 labels using machine learning kernel methods! This challenge is part of our coursework for Machine Learning with kernel methods (by Michael Arbel, Julien Mairal, Jean-Philippe Vert and Alessandro Rudi) of the Master MVA. Our goal is to develop and apply machine learning models based on kernel methods to accurately classify images into one of ten predefined labels. During this challenge, we implemented several methods to classify images in 10 classes. Features extractors are computing using both handmade code and implemented libraries. Then, kernel methods such as kernel PCA and kernel SVM are used to finally perform classification. We achieve a test score of 0.592 on the public leaderboard.

## How to run 

You can test several experiments by using the train.py script and choosing the wanted argument. You can run the algorithm which provides us with our best submission file on the leaderboard with the following command:
   ```
   python start.py
   ```
Furthermore, the pipeline we used to test various settings and carry out various experiments is available in run_exps.py which takes a dictionary of experiments (in last_experiments.py for instance) and run them one after another. 

## Dataset 

The studied dataset consists in 5000 train images and 2000 in the test set. They are 500 images of each label. This is an example of a normalized image to classify:
<p align="center">
   [alt text](https://github.com/lucasgascon/kaggle-km/blob/main/sampled_image.png?raw=true)
</p>


## Contact Information

For any inquiries or further information, please feel free to reach out to us at:

- Lucas Gascon (MVA ENS Paris-Saclay)
- Hippolyte Pilchen (MVA ENS Paris-Saclay)
We can be reached at: forename.name@polytechnique.edu

## Acknowledgments

We would like to express our gratitude to the organizers of this Kaggle challenge for providing the opportunity to apply our knowledge and skills in image classification with machine learning kernel methods.


