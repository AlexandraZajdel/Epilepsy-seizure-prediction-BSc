''' Make comparison analysis between preictal and interictal data samples. 

Run script as follows:
python visualization_raw_data.py --cfg 'config_dir.<config_name>'
'''

import sys

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import mne

from utils_plots import (time_freq_comparison_plots, correlation_plot, 
                        fft_plot, distribution_comparison_plot, time_domain_subplots)
from utils_transforms import remove_DC_component

sys.path.append('../..')
from scripts.preprocessors.visualizing import psd_plot
from scripts.preprocessors.utils import (load_mat_file, 
                                        array_to_dataframe_converter, load_config)

sns.set_palette('Reds')

if __name__ == '__main__':
    CONFIG = load_config(script_descr='Visualize raw data.')

    # provide paths to specified files
    path_preictal = '../../data/raw/Pat1Train/Pat1Train_107_1.mat'
    path_interictal = '../../data/raw/Pat1Test/Pat1Test_17_0.mat'

    df_pre = array_to_dataframe_converter(load_mat_file(path_preictal))
    df_inter = array_to_dataframe_converter(load_mat_file(path_interictal))

    df_pre = remove_DC_component(df_pre)
    df_inter = remove_DC_component(df_inter)

    # time_domain_subplots(CONFIG, (df_pre, df_inter))
    time_freq_comparison_plots(CONFIG, (df_pre, df_inter), time_stop=600)

    for df in [df_pre, df_inter]:
        print(df.describe().round(2))
        correlation_plot(df, 'Correlation between channels')
        psd_plot(CONFIG, df, is_across_channel=True)
    
    fft_plot(CONFIG, df_pre, df_inter)
    distribution_comparison_plot(df_pre, df_inter)
