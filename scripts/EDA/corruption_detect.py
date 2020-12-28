""" Count corrupted data rows for each subject. 
Generate accumulated_percentages plots. 

Based on the notebook: 
Zamora-Martinez, F (2016) DropoutCounts [Source code] https://www.kaggle.com/pakozm/dropoutcounts
Licence: Apache 2.0

Run script as follows:
python corruption_detect.py --cfg 'config_dir.<config_name>'
"""

import os
from glob import glob
import concurrent.futures
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.io

sys.path.append('../../')
from scripts.preprocessors.utils import (
    load_mat_file,
    array_to_dataframe_converter,
    load_config,
)

from scripts.models.utils import get_class_from_labels_file

sns.set_style("darkgrid")


def dropout_counts_single_file(filepath):
    """ Count corrupted rows for a single file. """

    global INVALID_VALUE

    # load file
    mat_data = load_mat_file(filepath)
    df = array_to_dataframe_converter(mat_data)

    # count rows with invalid value
    try:
        invalid_rows_count = (df == INVALID_VALUE).all(axis=1).value_counts()[True]
    except KeyError:
        # if there is no invalid rows
        invalid_rows_count = 0

    # count percentage of invalid rows
    invalid_percent = invalid_rows_count / df.shape[0]

    basename = os.path.basename(filepath)
    # retrieve class number from filename
    class_num = get_class_from_labels_file(CONFIG, filepath)

    return basename, class_num, invalid_rows_count, invalid_percent

def get_public_test_files():
    public_list = [
        line.strip()
        for line in open(
            os.path.join(CONFIG.paths['labels_dir'], 'public_test_names.csv'), "r"
        )
    ]
    return public_list


def dropout_counts_all(data_dir, subjects):
    """ Count corrupted rows for all files. """

    all_subj_dict = {
        subj: pd.DataFrame(
            columns=["filename", "class", "invalid_counts", "invalid_percent"]
        )
        for subj in subjects
    }

    for subj in subjects:
        print("[INFO] Counting dropouts of", subj)

        # get all .mat files from directory
        files = sorted(glob(os.path.join(data_dir, subj, "*.mat")))
        # if subj is from testing mode: select only public test files
        if 'Test' in subj:
            public_list = get_public_test_files()
            files = [path for path in files if 
                    os.path.basename(path).split(".")[0] in public_list]

        # run dropout_counts_single_file function in parallel
        with concurrent.futures.ProcessPoolExecutor() as executor:
            processes = []
            for filepath in files:
                future = executor.submit(dropout_counts_single_file, filepath)
                processes.append(future)

            for future in concurrent.futures.as_completed(processes):
                all_subj_dict[subj].loc[len(all_subj_dict[subj])] = future.result()

    return all_subj_dict


def print_subject_summary(subj_name, subject_dataframe):
    """ Print how many files for each subject and each class are corrupted. """

    print(f"[INFO] Percentage of files to drop regarding threshold: {DROP_THRESHOLD}")

    for class_number in [0, 1]:
        class_dataframe = subject_dataframe[subject_dataframe["class"] == class_number]

        n_files = class_dataframe.shape[0]

        n_dropouts = np.sum(class_dataframe["invalid_percent"] >= DROP_THRESHOLD)
        percent_dropout = n_dropouts / n_files * 100.0

        print(
            f"[INFO] {subj_name} class: {class_number}: \
        {n_dropouts} of {n_files}, ({percent_dropout:.2f} %)"
        )


def verify_file_rejection(subject_dataframe):
    """Create a new 'save' column and store info about
    saving file for further computations."""

    subject_dataframe["save"] = subject_dataframe["invalid_percent"] < DROP_THRESHOLD
    return subject_dataframe


def compute_accumulated_percentages(subject_dataframe, class_number):
    """ Compute accumulated percentage of invalid rows. """

    global NUM_ROWS

    class_dataframe = subject_dataframe[subject_dataframe["class"] == class_number]
    uniq_values, uniq_counts = np.unique(
        class_dataframe["invalid_counts"], return_counts=True
    )

    # get the inverse dropout values for visualization purposes
    inv_rows_dropout_percent = uniq_values[::-1] / NUM_ROWS
    inv_file_dropout_percent = np.cumsum(uniq_counts[::-1]) / float(np.sum(uniq_counts))
    return inv_rows_dropout_percent, inv_file_dropout_percent


def plot_percentages(ax, title, x, y, color, label):
    """ Create step subplot. """

    # convert to percentage
    x *= 100
    y *= 100

    ax.set(xlabel="% of corrupted rows", ylabel="% of files")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.invert_xaxis()
    ax.step(x, y, linewidth=2, color=color, where="post", label=label)
    ax.legend()


def subject_analysis(subj_name, subject_dataframe):
    subject_dataframe = verify_file_rejection(subject_dataframe)
    print_subject_summary(subj_name, subject_dataframe)

    # Compute accumulated percentage of files with at more than % dropout rows
    x_0, y_0 = compute_accumulated_percentages(subject_dataframe, class_number=0)
    x_1, y_1 = compute_accumulated_percentages(subject_dataframe, class_number=1)

    # extract corrupted files names
    files_to_drop = subject_dataframe[subject_dataframe["save"] == False]["filename"]

    return files_to_drop, (x_0, y_0), (x_1, y_1)


def generate_figure(data):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    colors = ["r", "g", "b"]

    color_idx = 0
    for (
        subj_name,
        row_pct_0_class,
        files_pct_0_class,
        row_pct_1_class,
        files_pct_1_class,
    ) in data:
        plot_percentages(
            ax1,
            "interictal",
            row_pct_0_class,
            files_pct_0_class,
            color=colors[color_idx],
            label=subj_name,
        )
        plot_percentages(
            ax2,
            "preictal",
            row_pct_1_class,
            files_pct_1_class,
            color=colors[color_idx],
            label=subj_name,
        )
        color_idx += 1

    plt.savefig(os.path.join(CONFIG.paths["results_dir"], "corruption_analysis.png"))


if __name__ == "__main__":
    CONFIG = load_config(script_descr="Corruption analysis.")
    subjects = ["Pat1Train", "Pat2Train", "Pat3Train"]
    data_path = CONFIG.paths["raw_data_dir"]

    if not os.path.exists(CONFIG.paths["results_dir"]):
        os.makedirs(CONFIG.paths["results_dir"])

    INVALID_VALUE = CONFIG.signal_params["invalid_value"]
    NUM_ROWS = 10 * 60 * 400
    # drop files with >= % of corrupted rows
    DROP_THRESHOLD = CONFIG.preprocessor["corruption_threshold"]

    all_subj_dict = dropout_counts_all(data_path, subjects)
    drop_df_buffer = []
    plot_data_buffer = []

    for subj in subjects:
        files_to_drop, (x_0, y_0), (x_1, y_1) = subject_analysis(
            subj, all_subj_dict[subj]
        )
        drop_df_buffer.append(files_to_drop)
        plot_data_buffer.append((subj, x_0, y_0, x_1, y_1))

    # generate plots
    generate_figure(plot_data_buffer)

    # if training sets: save filenames to drop (from all subjects) 
    if all([item.endswith('Train') for item in subjects]):
        pd.concat(drop_df_buffer).to_csv(
            os.path.join(CONFIG.paths["results_dir"], "files_to_drop.csv"),
            index=False,
            header=False,
        )
