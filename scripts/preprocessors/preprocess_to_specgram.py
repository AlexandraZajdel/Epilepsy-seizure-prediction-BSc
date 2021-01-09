'''Data preprocessing using FFT and generating binned spectrograms . 

Run script as follows:
python preprocess_to_specgram.py --cfg 'config_dir.<config_name>'
'''

import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, spectrogram

sys.path.append('../../')
from scripts.preprocessors.visualizing import (time_domain_plot, psd_plot,
                                                spectrogram_binned_plot_subplots)
from scripts.preprocessors.utils import (load_mat_file, run_preprocessor, load_config)


def apply_bandpass_filter(data, order=5):
    nyquist_freq = 0.5 * CONFIG.signal_params['sampling_freq']
    low = CONFIG.preprocessor['low_cut'] / nyquist_freq
    high = CONFIG.preprocessor['high_cut'] / nyquist_freq
    numerator, denominator = butter(order, [low, high], btype='band')
    # axis=0 in lfilter: filter down each column
    filtered_data = lfilter(numerator, denominator, data, axis=0)  
    return np.float32(filtered_data)


def group_into_bands(signal_fft, fft_freq):
    ''' Compute specified metric from fft amplitude values within 
    frequency bands defined in the configuration file. '''

    # convert values from brain activity bands dictionary
    # to list with unique values
    bins = []
    for ranges in CONFIG.preprocessor['brain_activ_bands'].values():
        for elem in ranges:
            if elem not in bins:
                bins.append(elem)

    freq_bands = np.digitize(fft_freq, bins)
    band_df = pd.DataFrame({'fft': signal_fft, 'band': freq_bands})

    function_mappings = {
        'mean': np.mean,
        'std': np.std,
        'max': np.max,
        'min': np.min,
    }

    try:
        # from config file get the metric to compute for each band
        metric_func = function_mappings[
            CONFIG.preprocessor['spectrogram']['metric']]
        band_df = band_df.groupby('band').apply(metric_func)
    except KeyError:
        print(f'Invalid metric function for binned spectrogram. \
            Choose from {function_mappings.keys()}')
        sys.exit(1)

    band_df_valid = band_df.fft[1:-1]  # remove edge artifacts
    return band_df_valid


def compute_fft_within_bands(data, epsilon=sys.float_info.epsilon):
    '''
    Args:
        data (numpy array): array with shape (num_timepoints, num_channels)

    Returns:
        numpy array of aggregated fft values with shape 
        (num_channels, num_freq_bands, num_time_frames)
    '''

    sampling_freq = CONFIG.signal_params['sampling_freq']
    time_frame_len = CONFIG.preprocessor['spectrogram']['time_frame']
    brain_activ_bands = CONFIG.preprocessor['brain_activ_bands']

    (data_len, n_channels) = data.shape
    data_len_sec = data_len // sampling_freq
    n_timesteps = data_len_sec // time_frame_len

    fft_aggregated = np.zeros(
        (n_channels, len(brain_activ_bands), n_timesteps))

    for chann_num in range(n_channels):
        fft_subdata = np.zeros((len(brain_activ_bands), n_timesteps),
                               dtype='float32')

        for frame_num, time_end in enumerate(
                range(0, data_len_sec - time_frame_len + 1, time_frame_len)):
            subdata = data[(time_end * sampling_freq) :(time_end + time_frame_len) \
            * sampling_freq, chann_num]

            # add epsilon to avoid dividing by zero in log10
            fft = np.log10(np.abs(np.fft.rfft(subdata)) + epsilon)
            fft_freq = np.fft.rfftfreq(n=subdata.shape[0],
                                       d=1.0 / sampling_freq)
            fft_subdata[:len(brain_activ_bands),
                        frame_num] = group_into_bands(fft, fft_freq)

        fft_aggregated[chann_num, :, :] = fft_subdata
    return fft_aggregated

def preprocess_file(filepath, is_plot=False):
    data = load_mat_file(filepath)
    data = apply_bandpass_filter(data)
    data_fft_bands = compute_fft_within_bands(data)
    if is_plot:
        print(f'[INFO] Creating plots for {os.path.basename(filepath)}')
        time_domain_plot(CONFIG, data)
        psd_plot(CONFIG, data, is_across_channel=True)
        spectrogram_binned_plot_subplots(CONFIG, data_fft_bands)

    # save preprocessed files
    new_filename = os.path.basename(filepath).replace('.mat', '.npy')
    subject_name = filepath.split(os.path.sep)[-2]
    time_frame = CONFIG.preprocessor['spectrogram']['time_frame']
    metric = CONFIG.preprocessor['spectrogram']['metric']
    new_file_dir = os.path.join(CONFIG.paths['processed_data_dir'], 
                                f'binned_specgram_timeframe_{time_frame}_metric_{metric}',
                                subject_name)

    if not os.path.exists(new_file_dir):
        os.makedirs(new_file_dir)

    np.save(os.path.join(new_file_dir, new_filename), data_fft_bands)

if __name__ == '__main__':
    CONFIG = load_config(script_descr='Preprocess raw data to spectrograms.')
    run_preprocessor(CONFIG, preprocess_file, False)