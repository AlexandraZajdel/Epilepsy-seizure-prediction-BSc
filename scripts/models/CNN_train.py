
""" Data classification using binned spectrograms and Convolutional Neural Network. 

Run script as follows:
python CNN_train.py --cfg 'config_dir.<config_name>'
"""

import sys
from glob import glob
import os
import logging
import datetime

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from ConvNet import ConvNet
from CNN_optim import run_optimization

sys.path.append('../../')
from scripts.preprocessors import scaler
from scripts.models.utils import (
    load_npy_data,
    split_data,
    get_inputs_and_outputs,
    load_config
)

tf.get_logger().setLevel("ERROR")

def run_model(params, X_train, y_train, X_val, y_val, 
            is_summary=False, log_dir=None):
    model = ConvNet(params, n_classes=1, input_data_shape=X_train.shape)
    model.define_sequential_model()
    if is_summary: model.print_summary()
    model.compile_model()
    model.fit_model(X_train, y_train, X_val, y_val, callback_dir=log_dir)
    return model

def get_scores(model, X_test, y_test):
    train_score = model.get_train_score()
    val_score = model.get_val_score()
    test_score = model.evaluate_model(X_test, y_test)
    return (train_score, val_score, test_score)

def standardize_data_per_channel(data, axis=0):
    # TO DO: REFACTOR IT 
    buffer = []
    for img in data:
        # calculate per-channel means and standard deviations
        means = img.mean(axis=(0,1), dtype='float64')
        stds = img.std(axis=(0,1), dtype='float64')
        # per channels standarization of pixels
        img = (img-means) / stds
        buffer.append(img)

    arr = np.array(buffer)
    return arr

def run_workflow(logger=None, is_optim_mode=True):
    """ Prepare data for training and run model. """

    train_in, train_out, test_in, test_out = get_inputs_and_outputs(
        CONFIG,
        "binned_specgram",
        load_npy_data,
        ".npy",
        mode=CONFIG.training_settings["mode"],
    )

    log_dir = os.path.join(
            CONFIG.paths["results_dir"],
            "logs",
            "CNN",
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )

    # separate validation data from training data
    X_train, X_val, y_train, y_val = split_data(CONFIG, np.array(train_in), 
                                                np.array(train_out))
    X_test, y_test = np.array(test_in), np.array(test_out)

    del (train_in, train_out, test_in, test_out) # free memory 

    if logger is not None:
        time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(
            f"""\n{time}
                \nPreprocessed data loaded. 
                Train num. examples, num. channels, num. frequency bins, num. timesteps: {X_train.shape}
                Val num. examples: {X_val.shape[0]}
                Test num. examples: {X_test.shape[0]}
                \nTraining mode: {CONFIG.training_settings['mode']}
                \nParameters spectrogram: 
                {CONFIG.preprocessor['spectrogram']}"""
        )

    # apply scaler ( fit and transform on train data; transform validation and test sets )
    X_train, scalers_train = scaler.scale_across_time(x=X_train, x_test=X_test)
    X_val, _ = scaler.scale_across_time(x=X_val, x_test=X_test, scalers=scalers_train)
    X_test, _ = scaler.scale_across_time(X_test, x_test=None, scalers=scalers_train) 

    # make channel last
    X_train, X_val, X_test = [
        np.moveaxis(data, 1, 3) for data in [X_train, X_val, X_test]
    ]


    if is_optim_mode:
        params = run_optimization(run_model, X_train, y_train, X_val, y_val)
    else:
        params = CONFIG.models['CNN']

    if logger is not None:
        logger.info(f'\nParameters CNN:\n{params}')

    # run training multiple times because of non-determinism
    score_list = []
    for _ in range(CONFIG.models["CNN"]["max_iterations"]):
        model = run_model(params, X_train, y_train, X_val, y_val, log_dir=log_dir)
        score_list.append(get_scores(model, X_test, y_test))

    scores_train, scores_val, scores_test = zip(*score_list)
    mean_score_train = np.mean(scores_train)
    mean_score_val = np.mean(scores_val)
    mean_score_test = np.mean(scores_test)

    if logger is not None:
        logger.info(f"\n{time}: Mean AUC score is: train: {mean_score_train:.2f} "+
                    f"val: {mean_score_val:.2f} " +
                    f"test: {mean_score_test:.2f}\n{'>'*80}")

    return mean_score_test


if __name__ == "__main__":
    CONFIG = load_config(script_descr="CNN training using customized configuration.")

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(CONFIG.paths["results_dir"], "CNN_model_logs.txt"),
        level=logging.INFO,
    )
    logger.setLevel(logging.INFO)

    mean_score = run_workflow(logger, is_optim_mode=False)

