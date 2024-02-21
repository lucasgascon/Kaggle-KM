import argparse
import numpy as np
from experiments import experiments
from train import main as train_main
from train import parser_args as train_parser_args
import json
    
def create_commands(hog, raw, lbp, PCA, kernelPCA, strat, SVM, kernelSVM, C, 
                    sigma, p, subname, gamma, r, bow, k, sift, fishervect, with_norm):
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
    return commands
    
def run_exps():
    accuracies = {}
    i = 0
    file_path = 'experiment_results.json'
    for num_exp, params in experiments.items():
        print('Running experiment', num_exp)
        parser = argparse.ArgumentParser()
        parser = train_parser_args(parser) 
        params['subname'] = f'exp_{num_exp}'
        args = parser.parse_args(create_commands(**params))
        print('Command:', args)
        accuracy = train_main(args)
        print('Accuracy:', accuracy)
        accuracies[num_exp] = accuracy
        if num_exp %20 == 0:
            # Save the dictionary to a JSON file
            with open(f'experiment_results_{i}.json', 'w') as f:
                json.dump(accuracies, f)
            i +=1
        print('Finished experiment', num_exp)
        print('')
        
    # Save the dictionary to a JSON file
    with open(file_path, 'w') as f:
        json.dump(accuracies, f)
        
if __name__ == "__main__":
    # Example usage
    run_exps()
    print("All experiments run successfully")
    