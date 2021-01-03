''' Function for optimization of Convolutional Neural Network 
using hyperopt library. '''

import logging

import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

logging.getLogger('hyperopt.tpe').setLevel(logging.ERROR)

def get_scores(model, X_test, y_test):
    train_score = model.get_train_score()
    val_score = model.get_val_score()
    test_score = model.evaluate_model(X_test, y_test)
    return (train_score, val_score, test_score)

def run_optimization(get_model_func, X_train, y_train, X_val, y_val, X_test, y_test):
    search_space = {
        'epochs': hp.choice('epochs', [100]), 
        'batch_size': hp.choice('batch_size', [8, 16]),
        'learning_rate': hp.choice('learning_rate', [0.001, 0.0001, 0.01]),
        'l2_reg': hp.choice('l2_reg', [0.01, 0.001, 0.1]),
        'nfilters_conv1': hp.choice('nfilters_conv1', [8, 16, 32, 64]),
        'nfilters_conv2': hp.choice('nfilters_conv2', [8, 16, 32, 64]),                                                
        'kernel_size_1': hp.choice('kernel_size_1', [3]),
        'kernel_size_2': hp.choice('kernel_size_2', [3]),
        'stride_conv1': hp.choice('stride_conv1', [1]),
        'stride_conv2': hp.choice('stride_conv2', [1]),
        'poolsize': hp.choice('poolsize', [2]),
        'poolstride': hp.choice('poolstride', [2]),
        'dropout': hp.uniform('dropout', 0.0, 0.8),                                            
        'dense_units': hp.choice('dense_units', [64, 128, 256]), 
    }

    def func_to_minimize(params):
        max_iterations = 10
        score_list = []

        # run training multiple times because of non-determinism
        for _ in range(max_iterations):
            model = get_model_func(params, X_train, y_train, X_val, y_val)
            score_list.append(get_scores(model, X_test, y_test))

        scores_train, scores_val, scores_test = zip(*score_list)
        mean_score_train = np.mean(scores_train)
        mean_score_val = np.mean(scores_val)
        mean_score_test = np.mean(scores_test)

        print(params)
        print(mean_score_train, mean_score_val, mean_score_test)
        # minimize the inverted test AUC score
        return {'loss': -mean_score_test, 'status': STATUS_OK}

    trials = Trials()
    best_params = fmin(func_to_minimize, 
                        search_space,
                        algo=tpe.suggest, 
                        max_evals=10,
                        trials=trials,
                        return_argmin=False)
    
    # return best params based on cross-validation score
    return best_params