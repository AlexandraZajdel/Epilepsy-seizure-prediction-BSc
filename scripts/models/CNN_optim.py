''' Function for optimization of Convolutional Neural Network 
using hyperopt library. '''

import logging

import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

logging.getLogger('hyperopt.tpe').setLevel(logging.ERROR)

def run_optimization(get_model_func, X_train, y_train, X_val, y_val):
    search_space = {
        'epochs': hp.choice('epochs', [150]), 
        'batch_size': hp.choice('batch_size', [8]),
        'learning_rate': hp.choice('learning_rate', [0.001]),
        'l2_reg': hp.choice('l2_reg', [0.01, 0.001, 0.1]),
        'nfilters_conv1': hp.choice('nfilters_conv1', [8, 16, 32, 64]),
        'nfilters_conv2': hp.choice('nfilters_conv2', [8, 16, 32, 64]),                                                
        'kernel_size_1': hp.choice('kernel_size_1', [3]),
        'kernel_size_2': hp.choice('kernel_size_2', [3]),
        'stride_conv1': hp.choice('stride_conv1', [1]),
        'stride_conv2': hp.choice('stride_conv2', [1]),
        'poolsize_conv1': hp.choice('poolsize_conv1', [2]),
        'poolsize_conv2': hp.choice('poolsize_conv2', [2]),
        'poolstride_conv1': hp.choice('poolstride_conv1', [2]),
        'poolstride_conv2': hp.choice('poolstride_conv2', [2]),
        'dropout': hp.uniform('dropout', 0.0, 0.8),                                            
        'dense_units': hp.choice('dense_units', [64, 128, 256]), 
    }

    def func_to_minimize(params):
        model = get_model_func(params, X_train, y_train, X_val, y_val)
        best_epoch_loss = np.argmin(model.history.history['val_loss'])
        best_val_loss = np.min(model.history.history['val_loss'])

        return {'loss': best_val_loss, 'best_epoch': best_epoch_loss, 
                'status': STATUS_OK, 'model': model, 'history': model.history}

    trials = Trials()
    best_params = fmin(func_to_minimize, search_space,
                        algo=tpe.suggest, 
                        max_evals=2,
                        trials=trials,
                        return_argmin=False)
    return best_params