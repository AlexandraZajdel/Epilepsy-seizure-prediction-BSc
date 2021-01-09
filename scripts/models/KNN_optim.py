''' Function for optimization of K-nearest neighbours algorithm 
using hyperopt library. '''

import logging

import numpy as np
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval

logging.getLogger('hyperopt.tpe').setLevel(logging.ERROR)

def run_optimization(get_clasifier_func, X_train, y_train):
    ''' Run K parameter optimization using hyperopt library. '''

    # define the search space over hyperparameters (only odd number of neighbors) 
    search_space = {
        'n_neighbors': hp.choice('n_neighbors', np.arange(9, 99, 2)),
    }

    def func_to_minimize(params):
        ''' Objective function.'''
        
        classifier = get_clasifier_func(params, X_train, y_train)
        auc = cross_val_score(classifier, X_train, y_train, 
                              scoring='roc_auc', cv=3, n_jobs=-1).mean()

        # goal is to maximize cross-validation auc score 
        # return the negative of this value for the use of hyperopt 
        return {'loss': -auc, 'status': STATUS_OK}

    trials = Trials()
    best_params = fmin(func_to_minimize, 
                       search_space, 
                       algo=tpe.suggest, 
                       max_evals=10, 
                       trials=trials,
                       return_argmin=False)

    # return best params based on cross-validation score
    return best_params