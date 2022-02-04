from transformer import CNNVIT as vit
import os
import json
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from torchsummary import summary
from helper import *
from torchvision import transforms
from baseline_simple import C3D as bl
#input shape : data format (sample, band, time, height, width)
X = t.rand(1, 3, 32, 224, 224)
test = bl(time=32)
# test = vit()
#summary(test, (100, 3, 6, 224, 224))
res = test(X)
print(res[0].shape)
print(res[1].shape)
#test.describe()


#create band and times arrays
timepoints = 6
t_step = int(36 / timepoints)
times = range(0,36,t_step)
bands = ["GRN", "NIR", "RED"]
"""
mean, std_dev = get_mean_std(times)
print(mean.shape)
print(std_dev.shape)

print(mean)
print(std_dev)

mean = [2.2148, 7.9706, 2.2510]
std = [ 1021.1434, 11697.6494,  1213.0621]
data_transform = transforms.Compose([
    ToTensor(),
    transforms.Normalize(mean, std, inplace=False)
])
"""
# Read in pre-processed dataset
# data format (sample, band, time, height, width)


