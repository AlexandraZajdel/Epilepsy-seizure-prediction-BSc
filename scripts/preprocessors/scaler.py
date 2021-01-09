import numpy as np
from sklearn.preprocessing import StandardScaler

def scale_across_time(x, x_test=None, scalers=None):
    ''' Standardize data across time. 

    Based on the script: 
    Korshunova, I (2014) data_scaler
    [Source code] https://github.com/IraKorshunova/kaggle-seizure-prediction/blob/master/utils/data_scaler.py
    Licence: MIT

    Flatten the spectrogram into a vector and from each value 
    substract the mean and divide by standard deviation calculated 
    over the complete train dataset.
    '''
    n_examples, n_channels, n_freq_bins, n_timesteps = x.shape

    if scalers is None:
        scalers = [None] * n_channels

    for i in range(n_channels):
        # change n_timesteps and n_freq_bins in place
        xi = np.transpose(x[:, i, :, :], axes=(0, 2, 1))
        xi = xi.reshape((n_examples * n_timesteps, n_freq_bins))

        if x_test is not None:
            xi_test = np.transpose(x_test[:, i, :, :], axes=(0, 2, 1))
            xi_test = xi_test.reshape((x_test.shape[0] * n_timesteps, n_freq_bins))
            xi_complete = np.vstack((xi, xi_test))
        else:
            xi_complete = xi

        if scalers[i] is None:
            scalers[i] = StandardScaler()
            scalers[i].fit(xi_complete)

        xi = scalers[i].transform(xi)

        xi = xi.reshape((n_examples, n_timesteps, n_freq_bins))
        xi = np.transpose(xi, axes=(0, 2, 1))
        x[:, i, :, :] = xi

    return x, scalers