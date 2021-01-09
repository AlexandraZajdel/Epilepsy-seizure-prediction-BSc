'''Utility funcions for data visualization.'''

import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import seaborn as sns

from utils_transforms import compute_fft

sns.set()

def time_domain_plot(time, signal, channel_num, type):
    ''' Plot signal in time domain. '''

    plt.plot(time, signal, c='b')
    plt.title('ch ' + str(channel_num) + f': {type}')
    plt.xlabel('Time[s]')
    plt.ylabel(r'Voltage [$\mu$V] ')

def time_domain_subplots(config, data_pair, time_stop=None):
    '''Generate time plots for preictal and interictal data.

    Args:
        data_pair (tuple): Tuple of Pandas Dataframes with 
                            preictal and interictal data correspondingly
        time_stop (float): Lower limit of time range in seconds
    '''

    n_samples, n_channels = data_pair[0].shape

    time_start = 0
    if time_stop is None:
        time_stop = n_samples * (1 / config.signal_params['sampling_freq'])

    time_vector = np.arange(start=time_start,
                            stop=time_stop,
                            step=(1 / config.signal_params['sampling_freq']))

    for channel in range(n_channels):
        plt.figure(figsize=(15, 12))

        plt.subplot(2, 1, 1)
        time_domain_plot(time_vector, data_pair[0].iloc[:len(time_vector),
                                                        channel], channel,
                         'preictal')

        plt.subplot(2, 1, 2)
        time_domain_plot(time_vector, data_pair[1].iloc[:len(time_vector),
                                                        channel], channel,
                         'interictal')
        plt.show()


def spectrogram_plot(config, data, channel_num, type, is_log=True):
    '''Generate spectrograms for preictal and interictal data.'''

    freq_min, freq_max = 1, 100
    sample_freqs, segment_times, spect_values = signal.spectrogram(
        data,
        fs=config.signal_params['sampling_freq'],
        scaling='spectrum',
        nperseg=config.preprocessor['spectrogram']['time_frame'] * config.signal_params['sampling_freq'],
        noverlap=config.preprocessor['spectrogram']['overlap_viz'])

    if is_log:
        spect_values = np.log10(spect_values)

    plt.pcolormesh(segment_times, sample_freqs, spect_values, cmap='jet')
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency [Hz]')
    plt.yscale('linear')
    plt.colorbar()
    plt.ylim(freq_min, freq_max)
    plt.title('ch ' + str(channel_num) + f': {type}')


def time_freq_comparison_plots(config, data_pair, time_stop=None):
    '''Generate time plots and spectrograms for preictal and interictal data. 

    Args:
        data_pair (tuple): Tuple of Pandas Dataframes with 
                            preictal and interictal data correspondingly
        time_stop (float): Lower limit of time range in seconds
    '''

    n_samples, n_channels = data_pair[0].shape

    time_start = 0
    if time_stop is None:
        time_stop = n_samples * (1 / config.signal_params['sampling_freq'])

    time_vector = np.arange(start=time_start,
                            stop=time_stop,
                            step=(1 / config.signal_params['sampling_freq']))

    for channel in range(n_channels):
        plt.figure(figsize=(15, 12))

        plt.subplot(2, 2, 1)
        time_domain_plot(time_vector, data_pair[0].iloc[:len(time_vector),
                                                        channel], channel,
                         'preictal')

        plt.subplot(2, 2, 2)
        time_domain_plot(time_vector, data_pair[1].iloc[:len(time_vector),
                                                        channel], channel,
                         'interictal')

        plt.subplot(2, 2, 3)
        spectrogram_plot(config, data_pair[0].iloc[:len(time_vector), channel],
                         channel, 'preictal')

        plt.subplot(2, 2, 4)
        spectrogram_plot(config, data_pair[1].iloc[:len(time_vector), channel],
                         channel, 'interictal')

        plt.show()


def correlation_plot(dataframe, title):
    '''Plot correlation between channels.

    Args:
        dataframe (pd.DataFrame): Data for 1 patient.
        title (str): Title of the plot.
    '''

    plt.figure(figsize=(10, 8))
    sns.heatmap(dataframe.corr(), vmin=-1, vmax=1, cmap='RdBu')
    plt.title(title)
    plt.show()


def fft_plot(config, dataframe_pre, dataframe_inter):
    ''' Compare each channel of preictal and interictal signal 
    in frequency domain. '''

    fig = plt.figure()

    for dataframe, color in zip([dataframe_pre, dataframe_inter], ['r', 'b']):
        for num, channel in enumerate(dataframe.columns):
            time_series = dataframe.loc[:, channel]
            freqs, magnitude = compute_fft(time_series.values, 
                                        config.signal_params['sampling_freq'])
            ax = fig.add_subplot(8, 2, num+1)
            ax.plot(freqs, magnitude, c=color, alpha=0.8)
            ax.set_yticks([])
            ax.set_title(f'Channel: {num}', fontdict={'fontsize': 8})
        
    plt.legend(['preictal', 'interictal'], loc='upper right')
    plt.xlabel('Frequency [Hz]')
    plt.show()

   
def distribution_comparison_plot(dataframe_pre, dataframe_inter):
    ''' Compare preictal and interictal signal using 
    Kernel Density Estimate plot with Gaussian kernels. '''

    for col1, col2 in zip(dataframe_pre.columns, dataframe_inter.columns):
        sns.kdeplot(dataframe_inter[col2], gridsize=500, color='b', legend=False)
        sns.kdeplot(dataframe_pre[col1], gridsize=500, color='r', legend=False)

    plt.legend(['interictal', 'preictal'])
    plt.xlim([-300, 300])
    plt.show()
