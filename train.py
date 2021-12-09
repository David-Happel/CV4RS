from enum import unique
import rasterio
import numpy as np
from rasterio.windows import Window
from matplotlib import pyplot
from pathlib import Path
import os
import re
import pandas as pd
import c3d
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader , TensorDataset, SubsetRandomSampler
from sklearn.model_selection import KFold
from torchsummary import summary

from baseline_simple import C3D as bl
from processdata import ProcessData
from helper import reset_weights
from helper import get_labels

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print(f'\nUsing gpu {torch.cuda.current_device()}')
else:
    print(f'\nUsing cpu')

# Change if need to process the data
process_data = False


#create band and times arrays
t_start = 1
t_stop = 37
t_step = 6
times = range(t_start,t_stop,t_step)
bands = ["GRN", "NIR", "RED"]
labels, label_names = get_labels()

#PREPARE DATA 
dl = ProcessData(bands = bands, times=times)

# Change if need to re-process the data
process_data = False

if process_data:
    #process training data 
    dl.process_tile("X0071_Y0043", out_dir = 'data/prepared/train/')
    #process test data 
    dl.process_tile("X0071_Y0043", out_dir='data/prepared/test/')

#create dataset
#data format (sample, band, time, height, width)

train_data, train_labels = dl.read_dataset(out_dir='data/prepared/train/')
test_data, test_labels = dl.read_dataset(out_dir='data/prepared/test/')
print(np.unique(train_labels))
#converting to tensor datasets
train_ds = TensorDataset(train_data , train_labels)
test_ds = TensorDataset(test_data , test_labels)

#TRAINING
criterion = nn.BCEWithLogitsLoss()

n_epochs = 1
k_folds = 5

kfold = KFold(n_splits=k_folds, shuffle=True)
val_acc = dict()
val_loss = dict()
for fold, (train_ids, test_ids) in enumerate(kfold.split(train_ds)):

    print("fold:", fold)
    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = SubsetRandomSampler(train_ids)
    test_subsampler = SubsetRandomSampler(test_ids)

    # Define data loaders for training and testing data in this fold
    train_batches = DataLoader(
                      train_ds, 
                      batch_size=10, sampler=train_subsampler)
    test_batches = DataLoader(
                      train_ds,
                      batch_size=10, sampler=test_subsampler)

    #model selection

    model = bl(bands=len(bands), labels=len(labels)).to(device)
    model.apply(reset_weights)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        #Feed the whole batch in and optimise over these samples
        for i, batch in enumerate(train_batches, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch
            inputs.to(device)
            labels.to(device)
            print(inputs.shape)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            print(running_loss)
            if i % 5 == 4:    # print every 4 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 5))
                running_loss = 0.0

    # Process is complete.
    print('Training process has finished. Saving trained model.')

    # Print about testing
    print('Starting testing')

    # Saving the model
    save_path = f'./models/saved/model-fold-{fold}.pth'
    torch.save(model.state_dict(), save_path)

    # Evaluationfor this fold
    correct, total = 0, 0
    with torch.no_grad():

        # Iterate over the test data and generate predictions
        for i, data in enumerate(test_batches, 0):
            # Get inputs
            inputs, targets = data
            inputs.to(device)
            targets.to(device)

            # Generate outputs
            outputs = model(inputs)

            # Set total and correct
            predicted = outputs.data.int()
            print(predicted)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            loss = criterion(outputs, targets)

            # Print accuracy
            print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
            print('--------------------------------')
            val_acc[fold] = 100.0 * (correct / total) / len(labels[1])
            val_loss[fold] = loss

# Print fold results
print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
print('Loss----------------------------')
sum = 0.0
for key, value in val_loss.items():
    print(f'Fold {key}: {value} %')
    sum += value
print(f'Average: {sum/len(val_acc.items())} %')
print('Accuracy----------------------------')
sum = 0.0
for key, value in val_acc.items():
    print(f'Fold {key}: {value} %')
    sum += value

print(f'Average: {sum/len(val_acc.items())} %')
#TEST EVALUATION
## do work on test set

