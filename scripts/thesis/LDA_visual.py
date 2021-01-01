""" Data visualization using Linear Discriminant Analysis (LDA).  

Run script as follows:
python LDA_visual.py --cfg 'config_dir.<config_name>'
"""

import sys
import os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

    plt.figure(figsize=(9,7), dpi=600)
    plt.bar(x=np.arange(n_components), height=exp_var, alpha=0.5, 
            label='Individual explained variance')
    plt.step(x=np.arange(n_components), y=cum_sum_exp_var, color='r', 
            where='mid', label='Cumulative explained variance')
    plt.xlabel('Principal component index')
    plt.ylabel('Explained variance ratio')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(CONFIG.paths['results_dir'], 'plots', 
                f'tSNE_explained_variance_{MODE}.png'), bbox_inches='tight')

def create_embeddings(data, target, plot_variance=False):
    ''' Apply PCA and LDA '''

    n_components_pca = 50
    pca = PCA(n_components=n_components_pca)
    data = pca.fit_transform(data)
    if plot_variance: plot_explained_var(n_components_pca, pca)

    model = LinearDiscriminantAnalysis()
    data_visual = model.fit_transform(data, target)

    return data_visual

def visualize_data_LDA_1D(data, colors, markers_type, markers_size):
    plt.figure(figsize=(9,2))
    
    for a,b,c,d in zip(data, colors, markers_type, markers_size):
        plt.scatter(a, np.ones(len(a)), c=b, s=d, marker=c, alpha=0.3)

    # for legend purposes
    plt.scatter(data[0], 1, c=colors[0], marker=markers_type[0], 
                s=100, alpha=0.3, label='train')
    plt.scatter(data[-1], 1, c=colors[-1], marker=markers_type[-1], 
                s=100, alpha=0.3, label='test')
    plt.legend(loc='upper right')
    plt.yticks([])
    plt.xlabel('LDA1')
    plt.tight_layout()
    plt.show()

def plot_distributions(data_train, data_test):
    ''' Compare train and test data distribution on the plot. '''

    plt.figure(figsize=(9,7), dpi=600)
    sns.distplot(data_train, hist=False, label='train', color='r', 
                kde_kws={"shade": True})
    sns.distplot(data_test, hist=False, label='test', color='b', 
                kde_kws={"shade": True})
    plt.legend(loc='upper left')
    plt.xlabel('Value')
    plt.ylabel('Density')

def run_workflow():
    """ Prepare data for visualization. """

    time_frame = CONFIG.preprocessor['spectrogram']['time_frame']
    metric = CONFIG.preprocessor['spectrogram']['metric']

    train_in, _, test_in, _ = get_inputs_and_outputs(
        CONFIG,
        f"binned_specgram_timeframe_{time_frame}_metric_{metric}",
        load_npy_data,
        ".npy",
        mode=MODE,
    )

    print('Preprocessed data loaded from directory: ' +
    f'binned_specgram_timeframe_{time_frame}_metric_{metric}')

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
    markers_type = ['o'] * len(X_train) + ['o'] * len(X_test)
    markers_size = [300] * len(X_train) + [100] * len(X_test)
    # two classes: train and test 
    target = ['train'] * len(X_train) + ['test'] * len(X_test)
    
    X_visual = create_embeddings(X_stacked, target)
    visualize_data_LDA_1D(X_visual, colors, markers_type, markers_size)


if __name__ == "__main__":
    CONFIG = load_config(script_descr="tSNE visualization.")
    MODE = CONFIG.training_settings["mode"]
    run_workflow()
