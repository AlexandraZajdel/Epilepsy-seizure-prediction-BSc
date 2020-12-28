""" This script contains functions of general utility used in various places 
throughout models scripts. """

import os
from glob import glob
import argparse
import importlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def get_command_line_arg(script_descr):
    parser = argparse.ArgumentParser(description=script_descr)
    required_arg = parser.add_argument_group("required arguments")
    required_arg.add_argument(
        "--cfg", required=True, type=str, help="configuration module name"
    )
    args = parser.parse_args()
    return args


def load_config(script_descr):
    args = get_command_line_arg(script_descr)
    config_path = args.cfg
    # dynamically load configuration file
    module = importlib.import_module(config_path, "../../")
    config = module.Configuration()
    return config

def get_class_from_labels_file(config, filepath):
    ''' Load label from file. '''

    basename = os.path.basename(filepath)
    filename = basename.split(".")[0]

    if 'Train' in filename:
        # extract label from filename
        class_num = int(filename[-1])
    elif 'Test' in filename:
        # load label from file 
        labels_path = os.path.join(config.paths['labels_dir'], 'test.csv')
        labels_data = pd.read_csv(labels_path, header=0)
        mask = (labels_data['image'] == filename)
        class_num = int(labels_data['class'][mask])
    else:
        raise NameError(f'Label for file: {filename} not found.')

    return class_num


def get_inputs_and_outputs(config, foldername, load_func, format, mode="all"):
    """ Load data for training: inputs and outputs. """

    get_data_file_names = lambda folder_type : glob(
            os.path.join(
                config.paths["processed_data_dir"],
                str(foldername),
                folder_type,
                "*" + str(format),
            )
        )

    if mode == "all":
        train_files = get_data_file_names('*Train')
        test_files = get_data_file_names('*Test')

    elif mode in ["Pat1", "Pat2", "Pat3"]:
        train_files = get_data_file_names(mode + '*Train')
        test_files = get_data_file_names(mode + '*Test')
    else:
        raise ValueError(
            "mode parameter during training should be: \
                        ['all', 'Pat1', 'Pat2', 'Pat3']"
        )

    # load data from file and class label
    train_inputs, train_outputs = zip(
        *[(load_func(path), get_class_from_labels_file(config, path)) 
        for path in train_files])

    test_inputs, test_outputs = zip(
        *[(load_func(path), get_class_from_labels_file(config, path)) 
        for path in test_files])

    return train_inputs, train_outputs, test_inputs, test_outputs


def load_npy_data(path):
    """ Load preprocessed data from .npy format. """

    data = np.load(path)
    return data


def load_csv_data(path):
    """ Load preprocessed data from .csv format. """

    data = pd.read_csv(path, header=0, index_col=0)
    return data


def split_data(config, X, y):
    """ Split data into train and validation sets according to provided ratio. """

    val_ratio = config.training_settings["data_split"]["val_ratio"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_ratio, shuffle=True, random_state=42
    )

    return X_train, X_val, y_train, y_val

def calc_neg_to_pos_ratio(classes):
    ''' Compute negative to positive class ratio. '''

    neg, pos = np.bincount(classes)
    ratio = neg / pos

    return ratio

def reshape_spectr_data_to_2D(x, y):
    ''' Reshape spectrogram data with shape 
    (n_examples, n_channels, n_freq_bands, n_time_frames) to 2D data with shape
    (n_examples * n_time_frames, n_channels * n_freq_bands). 
    Adjust output shape by reapeating labels. 
    '''
    
    n_examples, n_channels, n_bands, n_time_frames = x.shape
    x_new = np.zeros((n_examples * n_time_frames, n_channels, n_bands))
    for i in range(n_channels):
        xi = np.transpose(x[:, i, :, :], axes=(0, 2, 1))
        xi = xi.reshape((n_examples * n_time_frames, n_bands))
        x_new[:, i, :] = xi

    x_new = x_new.reshape((n_examples * n_time_frames, n_channels * n_bands))
    y_new = np.repeat(y, n_time_frames)
    return x_new, y_new
