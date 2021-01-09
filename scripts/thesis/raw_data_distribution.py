'''Illustrate distribution differences between 
raw training data and raw test data. (Figures: 4.4, A.1)
'''

import os
import sys
from glob import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

sys.path.append('../../')
from scripts.preprocessors.utils import (load_mat_file, load_config)

plt.style.use('seaborn')

def compute_histogram_data(data_paths):
    ''' Compute the histogram data for every single file and average them. '''

    # same bins range for all files 
    bins = np.arange(-200, 200, step=2)  
    hist_buffer = []

    for path in data_paths:
        try:
            data = load_mat_file(path)
            hist, bin_edges = np.histogram(data, bins=bins, density=True)
            hist_buffer.append(hist)
        except OSError:
            print(f'Invalid file: {path}')

    hist_buffer = np.array(hist_buffer)
    hist_avg = np.mean(hist_buffer, axis=0) 
    return bin_edges, hist_avg

def remove_invalid_values_bar(bin_edges, hist_data):
    ''' Remove bar at x=0 because of the occurrence of invalid value. '''

    bin_edges = list(bin_edges)
    hist_data = list(hist_data)

    zero_idx = bin_edges.index(0)
    bin_edges.pop(zero_idx)
    hist_data.pop(zero_idx)
    return bin_edges, hist_data

def smooth_data(data):
    ''' Smooth data using convolution. '''

    return np.convolve(data, np.ones(5)/5, mode='same')

def plot_distribution(bin_edges, hist_train, hist_test):
    ''' Plot distribution comparison between train and test data. '''

    hist_train = smooth_data(hist_train)
    hist_test = smooth_data(hist_test)

    plt.figure(figsize=(9,7), dpi=600)

    plt.plot(bin_edges[:-1], hist_train, label='train', color='b')
    plt.fill_between(bin_edges[:-1], hist_train, alpha=0.5, color='b')
    plt.plot(bin_edges[:-1], hist_test, alpha=0.6, label='test', color='r')
    plt.fill_between(bin_edges[:-1], hist_test, alpha=0.5, color='r')
   
    plt.legend(loc='upper right')
    plt.xlabel(r'Amplitude $[\mu V]$')
    plt.ylabel('Density')
    plt.savefig(os.path.join(CONFIG.paths['results_dir'], 
                'plots', 
                f'raw_data_distplot_{MODE}.png'),
                bbox_inches='tight')

def run_workflow(config):
    ''' Run analysis on the whole dataset or for specific patient. '''

    get_data_file_names = lambda folder_type : glob(
            os.path.join(
                config.paths['raw_data_dir'],
                folder_type,
                '*.mat'
            )
        )

    if MODE == 'all':
        train_files = get_data_file_names('*Train')
        test_files = get_data_file_names('*Test')

    elif MODE in ['Pat1', 'Pat2', 'Pat3']:
        train_files = get_data_file_names(MODE + '*Train')
        test_files = get_data_file_names(MODE + '*Test')
    else:
        raise ValueError(
            "mode parameter during training should be: \
                        ['all', 'Pat1', 'Pat2', 'Pat3']"
        )

    bin_edges, hist_train = compute_histogram_data(train_files)
    bin_edges, hist_train = remove_invalid_values_bar(bin_edges, hist_train)
    bin_edges, hist_test = compute_histogram_data(test_files)
    bin_edges, hist_test = remove_invalid_values_bar(bin_edges, hist_test)

    plot_distribution(bin_edges, hist_train, hist_test)
    

if __name__ == '__main__':
    CONFIG = load_config(script_descr='Plot raw data distributions.')
    MODE = CONFIG.training_settings['mode']
    run_workflow(CONFIG)