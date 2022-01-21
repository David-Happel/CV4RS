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

#input shape : data format (sample, band, time, height, width)
X = t.rand(100, 3, 6, 224, 224)
test = vit()
#summary(test, (100, 3, 6, 224, 224))
test(X)


