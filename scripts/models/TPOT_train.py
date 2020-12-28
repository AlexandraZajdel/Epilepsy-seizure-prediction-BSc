""" Data classification using Auto ML (TPOT library). 

Run script as follows:
python tpot_train.py --cfg 'config_dir.<config_name>'
"""
import sys
from glob import glob
import os

from sklearn.preprocessing import StandardScaler
import numpy as np
from tpot import TPOTClassifier

sys.path.append('../../')
from scripts.models.utils import (
    load_npy_data,
    get_inputs_and_outputs,
    load_config,
    reshape_spectr_data_to_2D
)

def save_evaluated_indv(optimizer):
    ''' Save all pipelines that were evaluated during 
    the pipeline optimization process. 
    Format: Pipeline name: score
    '''

    results_dict = optimizer.evaluated_individuals_

    file_indv_path = os.path.join(CONFIG.paths['results_dir'], 
                                'TPOT_indv_results.txt')
    with open(file_indv_path, 'w') as file:
        for key in results_dict:
            score = results_dict[key]['internal_cv_score']
            file.write(f'{key},{score:.4f}\n')

def run_optimization(X_train, y_train):
    ''' Optimize machine learning pipelines using genetic programming.'''
    
    pipeline_optimizer = TPOTClassifier(generations=None,
                                        max_time_mins=90,
                                        population_size=10, 
                                        cv=3,
                                        random_state=42, 
                                        verbosity=2, 
                                        scoring='roc_auc',
                                        n_jobs=-1,
                                        use_dask=False,
                                        config_dict='TPOT light')
    pipeline_optimizer.fit(X_train, y_train)
    pipeline_optimizer.export(os.path.join(CONFIG.paths["results_dir"], 
                                            'tpot_digits_pipeline.py'))
    save_evaluated_indv(pipeline_optimizer)
    return pipeline_optimizer


def run_workflow():
    """ Prepare data for training and run model. """

    train_in, train_out, _, _ = get_inputs_and_outputs(
        CONFIG,
        "binned_specgram",
        load_npy_data,
        ".npy",
        mode=CONFIG.training_settings["mode"],
    )

    X_train, y_train = np.array(train_in), np.array(train_out)

    del (train_in, train_out) # free memory 

    X_train, y_train = reshape_spectr_data_to_2D(X_train, y_train)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    run_optimization(X_train, y_train)


if __name__ == "__main__":
    CONFIG = load_config(script_descr="KNN training using customized configuration.")
    run_workflow()