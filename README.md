# Crop Classification from Sentinel-2 Time Series Data

David Happel

Stephen Moran

Anastasia Schlegel

## Overview

This repository contains code relating to the 'Project Computer Vision for Remote Sensing' course at Technische Universität Berlin. This project involved researching state of the art techniques for classifying crops from satellite imagery. Three different types of architechtures were investigated: a 3-D CNN (Baseline Model), a 2DCNN-LSTM hybrid model and a 2DCNN-Transformer hybrid model.

## 1 - Data

### 1.1 - Dataset

The dataset for this project contains Sentinel-2 covering the Brandenburg region, Germany.

- 13 tiles - each tile is a 3000x3000 pixel image

- 8 spectral bands of different spatial resolution
  - 10m: GRN, RED, NIR
  - 20m: RE (3x), SW (2x) - upsampled to 10m resolution
- Temporal resolution: sample every 10 days
- 36 time frames across 1 year
- 20 classes (crop species) - 6 most prevelant used in analysis

### 1.2 - Pre-Processing

Images are cropped into smaller patches with a sliding window approach:

- Window size: 224x224
- 24 pixel overlap
- 225 image patches

- 4D array for each sample - bands, time, width, height

processdata.py contains all the proprocessing code for the image data

## 2 - Models

#### 2.1 - 3D-CNN (Baseline)

![alt text](/images/3DCNN.png)

#### 2.2 - 2DCNN (Common)

![alt text](/images/2DCNN.png)

#### 2.3 - 2DCNN-LSTM

![alt text](/images/LSTM.png)

#### 2.4 - 2DCNN-Transformer

![alt text](/images/Trans.png)

## 3 - Evaluation

For each run, a folder is created under /reports. This folder will contain a full log of the console output, the saved model, and .npy's of the evaluation data.

The evaluation data is generated after every epoch on the training and validation set. In the end evaluation is also performed on on the testing set.

The mode outputs are evalutated using using the evaluation method in helper.py.

Several metrics are collected, including f1 score, precision, recall, Hamming loss, and EMR. The former metrics are collected using several averaging strategies, micro, macro, wieghted, and samples average (advisable for multilabel).

### 3.1 - Tensorboard

To view and visualise the performance of the models, tensorboard can be used. When a model is trained information realted to it is automatically exported to the '/runs' folder.

Run the following command to launch tensorboard and then navigate to http://localhost:6006/ to view

```
tensorboard --logdir=runs
```

## 4 - Experiments

The following experiments were conducted on these models during our research.

- Effect of different temporal features (timepoints)
- Effect of different satellite bands of performance
- Model complexity and computational performance

## 5 - Getting Started

#### 5.1 - Cloning Repo

```
git clone https://github.com/David-Happel/CV4RS.git
```

#### 5.2 - Installing requirements

```
cd CV4RS
pip install -r requirements.txt
```

To run one of the 3 models use the following command in the terminal

```
python main.py --samples 5 --epochs 2 --batch_size 5  --timepoints 6 --model bl --name run_name --no_process_data
```

Arguments:

- --samples: if you want to limit the numebr of samples the model is trained on you can specify that here. Leave empty for full dataset
- --epochs: how many epochs to train the model on. Where one epoch is the full set of data
- --batch: Specify the batch size to be used in training
- --timepoints: The number of temporal features to use. Max is 36
- --model: The model to be trained (bl, trans, lstm)
- --name: the name of the output run file. Can be imported into tensorboard later. Also names the report file.
- --lstm_layers: the amount of lstm layers if using lstm model.
- --trans_layers: the amount of transformer layers if using lstm model.
- --no_process_data: Prevents the data being preprocessed. Leave this out the first time you run a model.

## 6 - Repo Structure

```
.
├── README.md
├── arg_parser.py # Argument parser from command line
├── baseline_simple.py # 3D CNN model used as baseline
├── transformer.py # 2DCNN-Transformer hybrid model
├── CNN_LSTM_V4.py  # 2DCNN-LSTM hybrid model
├── data # Dataset(Not included in repo)
│   ├── deepcrop
│   │   └── tiles # Raw data before preprocessing
│   └── prepared  # Processed data tiles
├── dataset.py # Dataset class
├── helper.py # Helper functions
├── main.py # Core code - where models are trained and tested
├── main_cv.py # Main with cross validation. Deprecated
├── processdata.py # Class for data proprocessing
├── report.py # Class for reporting
├── requirements.txt # Requirements
├── runs # Directory where data for tensorboard saved
├── scripts # bash scripts for running different types of experiments
└── test.py # testing models before training
```
