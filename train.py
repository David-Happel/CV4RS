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

bands = ["GRN", "NIR", "RED"]

data_dir = "data/prepared/"

imageWidth = 224
imageHeight = 224
time_n = 36

sample_n = 16

data = np.empty((sample_n, len(bands), time_n, imageWidth, imageHeight))

for i, file in enumerate(os.listdir(data_dir)):
    if i > sample_n: break
    regex = re.search('(\d*)_(.*).tif', file)
    if not regex : continue
    sample = int(regex.group(1))
    band = regex.group(2)
    band_i = bands.index(band)
    if band_i < 0: print("band not found!")
    with rasterio.open(os.path.join(data_dir, file)) as src:
        data[sample, band_i] = src.read()

# print(data.shape)
# pyplot.imshow(data[10,1,5,:,:] , cmap='pink')
# pyplot.show()
        

label_df = pd.read_csv(data_dir+"labels.csv",index_col= 0)
unique_labels = list(filter(lambda c: c not in ["image_id"], label_df.columns))
labels = label_df[unique_labels].to_numpy()[:sample_n]
print(unique_labels)
# print(labels[:5])

c = c3d.C3D(bands=3, labels=len(unique_labels))
c = c.float()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(c.parameters(), lr=0.001, momentum=0.9)

# sample_indices = np.array(range(sample_n))
# batches = np.array_split(sample_indices, 3)
data = torch.from_numpy(data).float()
labels = torch.from_numpy(labels).float()

print(data.shape, labels.shape)

dataset = TensorDataset(data , labels)
batches = DataLoader(dataset , batch_size = 5, shuffle=True)

for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
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
        if i % 5 == 4:    # print every 4 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 5))
            running_loss = 0.0

print('Finished Training')