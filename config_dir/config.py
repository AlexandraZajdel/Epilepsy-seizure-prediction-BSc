class Configuration:
    paths = {
        'root': '../',
        'raw_data_dir': '../../data/raw',
        'processed_data_dir': '../../data/processed',
        'labels_dir': '../../data/labels',
        'results_dir': '../../results',
    }


    signal_params = {
        'sampling_freq': 400,
        'invalid_value': 0.033691406
    }


    preprocessor = {
        'corruption_threshold': 1.0,
        'brain_activ_bands': {
            'delta': (0.1, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta': (12, 30),
            'lowgamma': (30, 70),
            'highgamma': (70, 180)
        },
        'low_cut': 0.1,
        'high_cut': 180,
        'spectrogram': {
            'time_frame': 30,
            'metric': 'mean',
            'overlap_viz': 0,                                                       # overlap only for visualization purposes 
        },
        'wavelet_transform': {
            'wavelet_type': 'db4',
        },
        'wavelet_scattering': {
            'T': 10*60*400,
            'J': 8,
            'Q': 12,
            'log_eps': 1e-6,
        },
    }

    training_settings = {
        'data_split': {
            'val_ratio': 0.2
        },
        'mode': 'all'
    }

    models = {
        'CNN': {
            'epochs': 50,
            'max_iterations': 1,
            'batch_size': 8,
            'learning_rate': 0.001,
            'l2_reg': 0.1,
            'nfilters_conv1': 4,
            'nfilters_conv2': 4,                                                  
            'kernel_size_1': 3,
            'kernel_size_2': 3, 
            'stride_conv1': 1,
            'stride_conv2': 1,
            'poolsize_conv1': 2,
            'poolstride_conv1': 2,
            'poolsize_conv2': 2,
            'poolstride_conv2': 2,
            'dropout': 0.4,                                            
            'dense_units': 64,                                                   
        },
        'XGB': {
            'n_estimators': 100,
            'max_depth': 3,
        }
    }