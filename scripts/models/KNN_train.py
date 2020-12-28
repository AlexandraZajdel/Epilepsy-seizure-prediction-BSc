""" Data classification using binned spectrograms and K-nearest neighbours. 

Run script as follows:
python KNN_train.py --cfg 'config_dir.<config_name>'
"""
import sys
from glob import glob
import os
import logging
import datetime

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import numpy as np

from KNN_optim import run_optimization
sys.path.append('../../')
from scripts.models.utils import (
    load_npy_data,
    get_inputs_and_outputs,
    load_config,
    reshape_spectr_data_to_2D
)

def run_classifier(params, X_train, y_train):
    ''' Train K-nearest neighbors classifier with specified parameters 
    or in optimization mode. '''

    clf = KNeighborsClassifier(**params)
    clf.fit(X_train, y_train)
    return clf

def calc_AUC(y, pred):
    fpr, tpr, _ = metrics.roc_curve(y, pred, pos_label=1)
    out = metrics.auc(fpr, tpr)
    return out

def make_prediction(model, X, y, n_examples, n_time_frames):
    ''' Make prediction on test data and return score. '''

    # get the prediction for class 1
    y_pred = model.predict_proba(X)[:, 1]
    y_pred = np.reshape(y_pred, (n_examples, n_time_frames))
    y_pred = np.mean(y_pred, axis=1)
    score = calc_AUC(y, y_pred)
    return score

def run_workflow(logger=None, is_optim_mode=False):
    """ Prepare data for training and run model. """

    train_in, train_out, test_in, test_out = get_inputs_and_outputs(
        CONFIG,
        "binned_specgram",
        load_npy_data,
        ".npy",
        mode=CONFIG.training_settings["mode"],
    )

    X_train, y_train, X_test, y_test = [np.array(data) for data in 
                                        [train_in, train_out, test_in, test_out]]

    del (train_in, train_out, test_in, test_out) # free memory 

    n_test_examples = X_test.shape[0]
    n_train_examples = X_train.shape[0]
    n_time_frames = X_train.shape[-1]

    y_train_original = y_train
    y_test_original = y_test

    X_train, y_train = reshape_spectr_data_to_2D(X_train, y_train)
    X_test, y_test = reshape_spectr_data_to_2D(X_test, y_test)

    if logger is not None:
        time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(
            f"""\n{time}
                \nPreprocessed data loaded. 
                Train num. examples x num. timesteps, num. channels x num. frequency bins: {X_train.shape}
                Test num. examples x num. timesteps: {X_test.shape[0]}
                \nTraining mode: {CONFIG.training_settings['mode']}"""
        )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if is_optim_mode:
        params = run_optimization(run_classifier, X_train, y_train)
    else:
        params = CONFIG.models['KNN']

    model = run_classifier(params, X_train, y_train)
    score_train = make_prediction(model, X_train, y_train_original, 
                                  n_train_examples, n_time_frames)
    score_test = make_prediction(model, X_test, y_test_original, 
                                n_test_examples, n_time_frames)

    if logger is not None:
        logger.info(f"Params: {params}" +
                    f"\n{time}: Mean AUC score is: train: {score_train:.2f} "+
                    f"test: {score_test:.2f}\n{'>'*80}")
    return score_test


if __name__ == "__main__":
    CONFIG = load_config(script_descr="KNN training using customized configuration.")

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(CONFIG.paths["results_dir"], "KNN_model_logs.txt"),
        level=logging.INFO,
    )
    logger.setLevel(logging.INFO)

    run_workflow(logger, is_optim_mode=True)
