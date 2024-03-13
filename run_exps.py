import argparse
import numpy as np
from last_experiments import experiments
from train import main as train_main
from train import parser_args as train_parser_args
from last_experiments import to_submit
import json


"""This file is used to run experiments in parallel.
"""


def create_commands(hog, raw, lbp, PCA, kernelPCA, strat, SVM, kernelSVM, C,
                    sigma, p, subname, gamma, r, bow, k, sift, fishervect, with_norm, normalize, to_submit=False, augmented=False,
                    startmode=False, **kwargs):
    commands = []
    if hog:
        commands.append('--hog')
    if raw:
        commands.append('--raw')
    if lbp:
        commands.append('--lbp')
    if PCA:
        commands.append('--PCA')
        commands.append(str(PCA))
    if kernelPCA:
        commands.append('--kernelPCA')
        commands.append(kernelPCA)
    if strat:
        commands.append('--strat')
        commands.append(strat)
    if SVM:
        commands.append('--SVM')
        commands.append(SVM)
    if kernelSVM:
        commands.append('--kernelSVM')
        commands.append(kernelSVM)
    if C:
        commands.append('--C')
        commands.append(str(C))
    if sigma:
        commands.append('--sigma')
        commands.append(str(sigma))
    if p:
        commands.append('--p')
        commands.append(str(p))
    if subname:
        commands.append('--subname')
        commands.append(subname)
    if gamma:
        commands.append('--gamma')
        commands.append(str(gamma))
    if r:
        commands.append('--r')
        commands.append(str(r))
    if bow:
        commands.append('--bow')
    if k:
        commands.append('--k')
        commands.append(str(k))
    if sift:
        commands.append('--sift')
    if fishervect:
        commands.append('--fishervect')
    if with_norm:
        commands.append('--with_norm')
    if normalize:
        commands.append('--normalize')
    if to_submit:
        commands.append('--to_submit')
    if augmented:
        commands.append('--augmented')
    if startmode:
        commands.append('--startmode')
    return commands


def run_exps():
    accuracies = {}
    i = 0
    file_path = 'all_experiment_results_not_handmade.json'
    for ind, (num_exp, params) in enumerate(experiments.items()):
        print('Running experiment', num_exp)
        parser = argparse.ArgumentParser()
        parser = train_parser_args(parser)
        params['subname'] = f'exp_{num_exp}'

        try:
            args = parser.parse_args(create_commands(**params))
            print('Command:', args)
            accuracy = train_main(args)
        except:

            print('Error in experiment', num_exp)

            print('Try with SGD instead of CVXOPT')
            try:
                params['SVM'] = 'SGD'
                args = parser.parse_args(create_commands(**params))
                print('Command:', args)
                accuracy = train_main(args)
            except:
                print('Error in experiment', num_exp)
                print('Already tried with both SVMs and failed')
                accuracy = 0

        print('Accuracy:', accuracy)
        accuracies[num_exp] = accuracy
        if ind % 20 == 0 and ind != 0:
            # Save the dictionary to a JSON file
            with open(f'experiment_results_{i}.json', 'w') as f:
                json.dump(accuracies, f)
            i += 1
        print('Finished experiment', num_exp)
        print('')

    # Save the dictionary to a JSON file
    with open(file_path, 'w') as f:
        json.dump(accuracies, f)


def create_submit_files(startmode=False):

    for ind, (num_exp, params) in enumerate(to_submit.items()):
        print('Running experiment', num_exp)
        parser = argparse.ArgumentParser()
        parser = train_parser_args(parser)
        if startmode:
            params['startmode'] = True
        params['subname'] = f'exp_bestsubmit_{num_exp}'
        args = parser.parse_args(create_commands(**params))
        print('Command:', args)
        train_main(args)


if __name__ == "__main__":
    # Example usage
    # run_exps()
    create_submit_files()
    print("All experiments run successfully")
