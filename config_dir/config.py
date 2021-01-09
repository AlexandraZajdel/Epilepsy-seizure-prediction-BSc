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
        'corruption_threshold': 0.8,
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
    }

    training_settings = {
        'data_split': {
            'val_ratio': 0.2
        },
        'mode': 'Pat1'
    }

    models = {
        'CNN': {
            'epochs': 100,
            'max_iterations': 1,
            'batch_size': 8,
            'learning_rate': 0.0001,
            'l2_reg': 0.1,
            'nfilters_conv1': 8,
            'nfilters_conv2': 8,                                                  
            'kernel_size_1': 3,
            'kernel_size_2': 3, 
            'stride_conv1': 1,
            'stride_conv2': 1,
            'poolsize': 2,
            'poolstride': 2,
            'dropout': 0.8,                                            
            'dense_units': 64,   
            'optim_mode': False,                                                
        },
        'KNN': {
            'params': {
                'n_neighbors': 39,
            },
            'optim_mode': False, 
        }
    }