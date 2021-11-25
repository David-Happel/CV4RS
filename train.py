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

from processdata import ProcessData

#filenames 
data_dir = "data/deepcrop/tiles/X0071_Y0043/"
data_filename = '2018-2018_001-365_HL_TSA_SEN2L_{band}_TSI.tiff'
out_dir = "data/prepared/"


#create band and times arrays
t_start = 1
t_stop = 37
t_step = 1
times = range(t_start,t_stop,t_step)
bands = ["GRN", "NIR", "RED"]

#prepare data
dl = ProcessData(data_dir, data_filename, out_dir)
#dl.prepare_data(times, bands)

#create dataset
data, labels = dl.create_dataset(t_samples=10)

data = torch.from_numpy(data).float()
labels = torch.from_numpy(labels).float()


print(data.shape, labels.shape)


#model selection
c = c3d.C3D(bands=3, labels=len(labels[1]))
c = c.float()

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(c.parameters(), lr=0.001, momentum=0.9)


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