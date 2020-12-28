""" Data visualization using t-distributed stochastic neighbor embedding.  

Run script as follows:
python tSNE_visual.py --cfg 'config_dir.<config_name>'


NOTES: the distribution of preprocessed training data as well as test data is the same. 
"""

import sys
import os

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sys.path.append('../../')
from scripts.models.utils import (
    load_npy_data,
    get_inputs_and_outputs,
    load_config,
)

plt.style.use('seaborn')

def reshape_to_2D(data):
    return np.reshape(data, (data.shape[0], data.shape[1]*data.shape[2]*data.shape[3]))

def plot_explained_var(n_components, pca):
    exp_var = pca.explained_variance_ratio_
    cum_sum_exp_var = np.cumsum(exp_var)

    plt.bar(x=np.arange(n_components), height=exp_var, alpha=0.5, 
            label='Individual explained variance')
    plt.step(x=np.arange(n_components), y=cum_sum_exp_var, color='r', 
            where='mid', label='Cumulative explained variance')
    plt.xlabel('Principal component index')
    plt.ylabel('Explained variance ratio')
    plt.legend(loc='upper right')
    plt.show()

def create_embeddings(data, plot=False):
    n_components_pca = 50
    pca = PCA(n_components=n_components_pca)
    data = pca.fit_transform(data)

    if plot: plot_explained_var(n_components_pca, pca)

    # take only 2 first components for visualization purposes
    model = TSNE(n_components=2, perplexity=40, learning_rate=100, 
                random_state=42)
    data_visual = model.fit_transform(data)
    return data_visual

def visualize_data(data, colors, markers):
    plt.figure(figsize=(9,7), dpi=600)
    
    for a,b,c,d in zip(data[:,0], data[:,1], colors, markers):
        plt.scatter(a, b, c=c, s=60, marker=d)

    plt.scatter(data[0, 0], data[0, 1], c=colors[0], marker=markers[0], 
                s=60, label='train')
    plt.scatter(data[-1, 0], data[-1, 1], c=colors[-1], marker=markers[-1], 
                s=60, label='test')
    plt.legend(loc='upper right')

    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.savefig(os.path.join(CONFIG.paths['results_dir'], 'plots', 
                f'tSNE_visual_{MODE}.png'), bbox_inches='tight')

def plot_distributions(data_train, data_test):
    plt.figure(figsize=(9,7), dpi=600)
    sns.distplot(data_train, hist=False, label='train', color='r', 
                kde_kws={"shade": True})
    sns.distplot(data_test, hist=False, label='test', color='b', 
                kde_kws={"shade": True})
    plt.legend(loc='upper left')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.savefig(os.path.join(CONFIG.paths['results_dir'], 'plots', 
                f'processed_data_distplot_{MODE}.png'), bbox_inches='tight')

def run_workflow():
    """ Prepare data for visualization. """

    train_in, _, test_in, _ = get_inputs_and_outputs(
        CONFIG,
        "binned_specgram",
        load_npy_data,
        ".npy",
        mode=MODE,
    )

    X_train, X_test = np.array(train_in), np.array(test_in)

    # reshaping to 2D
    X_train = reshape_to_2D(X_train)
    X_test = reshape_to_2D(X_test)

    plot_distributions(X_train, X_test)

    # since PCA determines the components based on variance,
    # it is mandatory to standardize the features before applying PCA
    X_stacked = np.vstack((np.float64(X_train), np.float64(X_test)))
    scaler = StandardScaler()
    X_stacked = scaler.fit_transform(X_stacked)

    colors = ['r'] * len(X_train) + ['b'] * len(X_test)
    markers = ['o'] * len(X_train) + ['^'] * len(X_test)

    X_visual = create_embeddings(X_stacked)
    visualize_data(X_visual, colors, markers)


if __name__ == "__main__":
    CONFIG = load_config(script_descr="tSNE visualization.")
    MODE = CONFIG.training_settings["mode"]

    run_workflow()
