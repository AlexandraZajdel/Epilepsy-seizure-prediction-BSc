''' This script contains functions of general utility used in various places 
throughout preprocessing scripts. '''

import glob
import os
import concurrent.futures
from tqdm import tqdm
import argparse
import importlib

import pandas as pd
from scipy.io import loadmat


def get_command_line_arg(script_descr):
    ''' Get configuration file parth from command line. '''

    parser = argparse.ArgumentParser(description=script_descr)
    required_arg = parser.add_argument_group("required arguments")
    required_arg.add_argument(
        "--cfg", required=True, type=str, help="configuration module name"
    )
    args = parser.parse_args()
    return args


def load_config(script_descr):
    ''' Load configuration file. '''

    args = get_command_line_arg(script_descr)
    config_path = args.cfg
    # dynamically load configuration file
    module = importlib.import_module(config_path, "../../")
    config = module.Configuration()
    return config


def load_mat_file(path):
    ''' Load .mat file and return numpy array with shape
    (NUM_SAMPLES, NUM_CHANNELS).'''

    mat_data = loadmat(path)
    return mat_data["data"]


def array_to_dataframe_converter(data):
    ''' Convert numpy array to pandas dataframe. '''

    df = pd.DataFrame(data=data)
    return df


def run_preprocessor(config, preprocess_function, is_parallel=True):
    ''' Run preprocess function on the whole dataset. '''

    data_dir = config.paths["raw_data_dir"]
    mode = config.training_settings["mode"]

    if mode == "all":
        file_paths = glob.glob(os.path.join(data_dir, "*", "*.mat"))
    elif mode in ["Pat1", "Pat2", "Pat3"]:
        file_paths = glob.glob(
            os.path.join(
                data_dir,
                mode + "*",
                "*.mat",
            )
        )
    else:
        raise ValueError(
            "mode parameter during training should be: \
                        ['all', 'Pat1', 'Pat2', 'Pat3']"
        )

    get_file_list = lambda filefolder, filename : [
        line.strip()
        for line in open(
            os.path.join(config.paths[filefolder], filename), "r"
        )
    ]

    # load the list of train files to drop and public test files
    drop_list = get_file_list('results_dir', 'files_to_drop.csv')
    public_test_list = get_file_list('labels_dir', 'public_test_names.csv')

    # filter train files: not in the drop_list
    file_paths_train = [path for path in file_paths if 
                        (('Train' in os.path.basename(path)) 
                        & (os.path.basename(path) not in drop_list))]
    # filter test files: only in public test
    file_paths_test = [path for path in file_paths if 
                        (('Test' in os.path.basename(path)) 
                        & (os.path.basename(path).split(".")[0] in public_test_list))]   

    # join file paths
    file_paths = [*file_paths_train, *file_paths_test]

    if not os.path.exists(config.paths["processed_data_dir"]):
        os.makedirs(config.paths["processed_data_dir"])

    if is_parallel:
        # run function in parallel
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(preprocess_function, file_paths)
    else:
        # serial function calling
        for path in tqdm(file_paths):
            preprocess_function(path)
