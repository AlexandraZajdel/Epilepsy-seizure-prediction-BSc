import numpy as np
import scipy

def remove_DC_component(data):
    return data - data.mean()


def compute_fft(signal, sampling_freq):
    '''
    Compute Fast Fourier Transform for one channel.

    Args:
        signal (array): 1D array of data for one channel
    '''
    delta_t = 1 / sampling_freq
    fft_values = scipy.fftpack.fft(signal)
    magnitude = np.abs(fft_values)
    freq = scipy.fftpack.fftfreq(sampling_freq,
                                 delta_t)

    positive_mask = np.where(freq > 0)
    freqs_to_plot = freq[positive_mask]
    magnitude_to_plot = magnitude[positive_mask]

    return freqs_to_plot, magnitude_to_plot
