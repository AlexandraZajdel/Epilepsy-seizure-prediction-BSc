# Epileptic seizures classification based on iEEG signals using machine learning techniques

This repository contains all the content related to my <b>Bachelor's Degree Final Project</b> at <i>AGH University of Science and Technology. </i>
The dissertation with a comprehensive description of used methods can be found 
[here](https://misio.fis.agh.edu.pl/media/misiofiles/8bca5a7fb32fc56aba98a469880af108.pdf). 

## Overview
The aim of the project is to predict epileptic seizure based on iEEG signal using classical Machine Learning as well as Deep Learning techniques.
The challenge is to distinguish between 10 min long data clips covering an hour prior to a seizure (**preictal state**), and 10 min iEEG clips of interictal activity (**interictal state**). The problem boils down to binary classification on imbalanced data. 

Description of the challenge in details: [Kaggle competition](https://www.kaggle.com/c/melbourne-university-seizure-prediction)

## Data 
The data is provided by Melbourne University and can be accessed via website https://www.epilepsyecosystem.org/. 
> Publication regarding NeuroVista Seizure Prediction Data collection: https://doi.org/10.26188/5b6a999fa2316

Once you have access to dataset, put MATLAB train and test data for each Patient into [data/raw](data/raw) folder. 

Test set labels and public test names can be updated in [data/labels](data/labels) folder.


## Project structure
The directories are listed below.
```bash
.
├── config_dir          # configuration files directory 
├── data 
│   ├── labels          # test set labels
│   ├── processed       # processed data files
│   └── raw             # original data files (.mat)
├── results
│   ├── logs            # TensorBoard logs for ConvNet
│   └── plots           # saved plots
└── scripts
    ├── EDA             # Exploratory Data Analysis scripts e.g. corruption detection
    ├── models          # proposed models for data classification
    ├── preprocessors   # preprocessing methods and features extraction
    └── thesis          # scripts to generate plots for dissertation
```

## Installation
The code was written in Python 3.8. Use the package manager **pip** to install requirements. Virtual environment is recommended. 
```bash
pip install -r requirements.txt
```

## Usage in Linux
1. Customize project configuration settings that are stored in [config.py](config_dir/config.py).

2. Find corrupted files:
```bash
cd scripts/EDA
python3 corruption_detect.py --cfg config_dir.config
```

Filenames to drop before further analysis will be saved in [result](result) directory. 

3. Preprocess raw data. 
- preprocess data:
```bash
cd scripts/preprocessors
python3 preprocess_to_specgram.py --cfg config_dir.config
``` 

Preprocessed data will be saved in [data/processed](data/processed) directory. 

4. Make classification

- train CNN model:
```bash
cd scripts/models
python3 CNN_train.py --cfg config_dir.config
``` 

- run K-nearest neighbors classifier:
```bash
cd scripts/models
python KNN_train.py --cfg config_dir.config
``` 

Results of classification are displayed in the terminal. 

## Optimization mode
CNN hyperparameters and KNN parameter K can be optimized using <b>Hyperopt</b> library. For this purpose change <i>'optim_mode'</i> to True in configuration file. 

Optimization is implemented in [CNN_optim.py](scripts/models/CNN_optim.py) and [KNN_optim.py](scripts/models/KNN_optim.py) with predefined search space.


## Author
Aleksandra Pestka

*Faculty of Physics and Applied Computer Science <br>
AGH UST University of Science and Technology*

*Cracow, January 2021*