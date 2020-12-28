# Epileptic seizures classification based on iEEG signals using machine learning techniques

This repository contains all the content related to my Bachelor's Degree Final Project. 
A comprehensive description can be found [here] (...). 

## Overview
The aim of the project is to predict epileptic seizure based on iEEG signal using classical Machine Learning as well as Deep Learning techniques.
The challenge is to distinguish between 10 min long data clips covering an hour prior to a seizure (**preictal state**), and 10 min iEEG clips of interictal activity (**interictal state**). The problem boils down to binary classification on imbalanced data. 

Description of the challenge in details: [Kaggle competition](https://www.kaggle.com/c/melbourne-university-seizure-prediction)

## Data 
Data is credited to Melbourne University and can be accessed via website https://www.epilepsyecosystem.org/. 
> Publication regarding collecting NeuroVista Seizure Prediction Data https://doi.org/10.26188/5b6a999fa2316


## Project structure

```bash
.
└── scripts
    ├── EDA             # Exploratory Data Analysis e.g. corruption detection
    ├── models          # proposed models for data classification
    ├── preprocessors   # preprocessing methods and features extraction
    └── thesis          # generating plots for dissertation
```

## Installation
The code was written in Python 3. Use the package manager **pip** to install requirements:
```bash
pip install -r requirements.txt
```

## Usage  
Project configuration settings are stored in [config_dir/config.py](config_dir/config.py).

1. Find corrupted files:
```bash
cd scripts/EDA
python3 corruption_detect.py --cfg config_dir.config
```

2. Preprocess raw data and classify. 
- preprocess data:
```bash
cd scripts/preprocessors
python3 preprocess_to_specgram.py --cfg config_dir.config
``` 

- Train CNN model:
```bash
cd scripts/models
python3 CNN_train.py --cfg config_dir.config
``` 

- Run KNN classificator:
```bash
cd scripts/models
python KNN_train.py --cfg config_dir.config
``` 


## Author
Aleksandra Pestka

*Faculty of Physics and Applied Computer Science <br>
AGH UST University of Science and Technology*

*Cracow, January 2021*