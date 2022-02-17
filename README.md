

# Crop Classification from Sentinel-2 Time Series Data

## Overview 
This repository contains code relating to the 'Project Computer Vision for Remote Sensing' course at Technische Universität Berlin. This project involved researching state of the art techniques for classifying crops from satellite imagery. Three different types of architechtures were investigated: a 3-D CNN, a 2DCNN-LSTM hybrid model and a 2DCNN-Transformer hybrid model. 

## Data 
The dataset for this project contains Sentinel-2 covering the Brandenburg region, Germany. 

8 spectral bands of different spatial resolution
- 10m: GRN, RED, NIR
- 20m: RE (3x), SW (2x)
- Temporal resolution: sample every 10 days 
- 36 time frames across 1 year 

- 20 classes (crop species) - 6 most prevelant used in analysis

## Dataset Preparation


- 3000x3000 pixel images

Images are cropped into smaller patches with a sliding window approach: 
- Window size: 224x224
- 24 pixel overlap
- 225 image patches

-  4D array for each sample - bands, time, width, height 


processdata.py contains all the proprocessing code for the image data 

## Experiments
The following experiments were conducted on these models during our research. 
- Effect of different temporal features (timepoints)
- Effect of different satellite bands of performance
- Model complexity and computational performance 



## Getting Started 

To run one of the 3 models use the following command in the terminal

```
python main.py --samples 5 --epochs 2 --batch_size 5  --timepoints 6 --model bl --name run_name --no_process_data
```

Arguments: 
- Samples: if you want to limit the numebr of samples the model is trained on you can specify that here. Leave empty for full dataset 
- Epochs: how many epochs to train the model on. Where one epoch is the full set of data
- Batch Size: Specify the batch size to be used in training 
- Timepoints: The number of temporal features to use. Max is 36 
- Model: The model to be trained (bl, trans, lstm)
- name: the name of the output run file. Can be imported into tensorboard later 
- --no_process_data: Prevents the data being preprocessed. Leave this out the first time you run a model. 


## HPC getting started

### Accessing server
Note: you have to be connected to the TUB network to access the HPC. Otherwise you have to use the TUB VPN. 

```
ssh <TUBID>@gateway.hpc.tu-berlin.de
```

### Downloading Repo
git clone

```
git clone git@github.com:David-Happel/CV4RS.git
```

### Python + virtual environment
Setting up the virtual environment on the server
Do this once and then it will be called in the bash script thereafter

```
mkdir venv
mkdir venv/cv4rs
module load python/3.7.1
python3 -m venv /home/users/d/davidhappel/venv/cv4rs
source /home/users/d/davidhappel/venv/cv4rs/bin/activate
```

### Installing requirements 

```
cd CV4RS
pip install -r requirements.txt
```

### Tranferring Data - Connecting via SSHFS
Mount the HPC to your local file directory when everything else is setup

Note: transferring takes a long time on home wifi. Would recommend using the TUB network

```
sshfs davidhappel@gateway.hpc.tu-berlin.de: <filepath to where you want to access the folder>
```

OOnce mounted you can then copy over data in your file explorer/finder
