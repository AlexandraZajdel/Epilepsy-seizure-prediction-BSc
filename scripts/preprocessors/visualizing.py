from mne.viz import plot_raw_psd
import mne 
import matplotlib.pyplot as plt


def create_rawarray(config, data):
    ''' Create Raw Array MNE object for visualization purposes. '''

    # define channel information
    ch_names = [f'ch{num}' for num in range(data.shape[1])]
    ch_types = ['ecog' for col in range(data.shape[1])]

    # create the info structure needed by MNE
    info = mne.create_info(ch_names=ch_names,
                           sfreq=config.signal_params['sampling_freq'],
                           ch_types=ch_types)

    # create the raw object
    raw = mne.io.RawArray(data.T, info)
    return raw


def time_domain_plot(config, data):
    ''' Easy time-domain plot using MNE library. 
    
    Args:
        data: numpy array with shape (NUM_TIMESTEPS, NUM_CHANNELS)
    '''

    raw = create_rawarray(config, data)

    # plot raw signal: set sensitivity and timebase
    raw.plot(n_channels=10,
             duration=10,
             scalings={'ecog': 2e2},
             title='Original data',
             show=True,
             block=True,
             color={'ecog': 'b'})


def psd_plot(config, data, is_across_channel=True):
    ''' Plot the power spectral density across channels or with averaging. 
    
    Args:
        data: numpy array with shape (NUM_TIMESTEPS, NUM_CHANNELS)
    '''

    raw = create_rawarray(config, data)
    if is_across_channel:
        plot_raw_psd(raw, xscale='linear', average=False, dB=True, color='m')
    else:
        plot_raw_psd(raw, xscale='linear', average=True, dB=True, color='m')


def spectrogram_binned_plot(config, data, channel_num, ax=None):
    ''' Spectrogram plot with calculated metric from fft amplitude 
    within predefined frequency bands. 
    
    Args:
        data: numpy array of aggregated fft values with shape 
            (NUM_CHANNELS, NUM_FREQ_BANDS, NUM_TIME_FRAMES)
        channel_num (int): channel index 
    '''

    if ax is None: ax = plt.gca()
    bands_names = config.preprocessor['brain_activ_bands'].keys()

    ax.set_yticks(range(len(bands_names)))
    ax.set_yticklabels(bands_names)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(10)
    ax.set_xticks(range(data.shape[2]))
    ax.set_xticklabels(range(data.shape[2]))

    plt.imshow(data[channel_num, :, :],
               aspect='auto',
               origin='lower',
               interpolation='none',
               cmap='jet')

    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=10)

    ax.set_title(f'Channel num: {channel_num}')
    ax.set_xlabel('Time frame', fontsize=10)


def spectrogram_binned_plot_subplots(config, data):
    fig = plt.figure(figsize=(15, 15))
    for channel_num in range(data.shape[0]):
        ax = fig.add_subplot(4, 4, channel_num + 1)
        spectrogram_binned_plot(config, data, channel_num, ax=ax)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    plt.show()
