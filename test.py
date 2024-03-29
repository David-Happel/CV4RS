
import os
import json
import numpy as np
from sklearn import naive_bayes
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from torchinfo import summary
from helper import *
from torchvision import transforms

#models
from baseline_simple import C3D as bl
from transformer import CNNVIT as vit
from CNN_LSTM_V4 import CNN_LSTM as cnn_lstm


#input shape : data format (sample, band, time, height, width)
samples = 10
bands = 3
times = 36
height = 224
width = 224
labels = 9

#text_file = open("complexity/{t}t_{b}b.txt".format(t = times, b = bands), "w")

X = t.rand(samples, bands, times, height, width)


bl = bl(bands, labels, times )
trans = vit(bands,labels, times)
lstm = cnn_lstm(bands, labels, times)

#TEST Models
l1, p1 = bl(X)
print(p1.shape)
l2, p2 = trans(X)
print(p2.shape)
l3, p3 = lstm(X)
print(p3.shape)



#COMPLEXITY 
print("====BASELINE=====\n")
model_sum = summary(bl, X.shape)

print("====TRANS=====\n")
model_sum = summary(trans, X.shape)

print("====LSTM=====\n")
model_sum = summary(lstm, X.shape)

#text_file.close()

"""
# test = vit()
#summary(test, (100, 3, 6, 224, 224))
res = test(X)
print(res[0].shape)
print(res[1].shape)
#test.describe()
"""


