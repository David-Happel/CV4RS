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
from torch.utils.data import DataLoader , TensorDataset
from torchsummary import summary
from baseline_simple import C3D as bl

from processdata import ProcessData

# Change if need to process the data
process_data = True

#create band and times arrays
t_start = 1
t_stop = 37
t_step = 6
times = range(t_start,t_stop,t_step)
bands = ["GRN", "NIR", "RED"]

#prepare data
dl = ProcessData(bands = bands, times=times)

if process_data:
    dl.process_tile("X0071_Y0043")

#create dataset
data, labels = dl.read_dataset()

#Splitting data
X_train, X_test, X_val, y_train, y_test, y_val = dl.train_test_val_split(data, labels, 0.2, 0.1)


# data format (sample, band, time, height, width)
data = torch.from_numpy(data).float()
labels = torch.from_numpy(labels).float()

print(data.shape, labels.shape)

#model selection
c = bl(bands=3, labels=len(labels[1]))
c = c.float()

#model summary
summary(c, (3, 6, 224, 224))


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(c.parameters(), lr=0.001, momentum=0.9)

"""
#Dataset Creation
dataset = TensorDataset(data , labels)
batches = DataLoader(dataset , batch_size = 5, shuffle=True)

for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    #Feed the whole batch in and optimise over these samples
    for i, batch in enumerate(batches, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = batch
        print(inputs.shape)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = c(inputs)
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

print('Finished Training')
"""