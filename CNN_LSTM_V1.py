#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as f
class CNN_LSTM(nn.Module):
    
    def __init__(self, bands = 3, labels = 19, device="cpu"):
        
        super(CNN_LSTM, self).__init__()

        self.device = device
        
        # default stride (1), padding (0), dilation (1)
        self.conv1 = nn.Conv2d(in_channels = bands, out_channels = 64, kernel_size = 3, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 2, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 2, padding = 1)
        
        self.pool1 = nn.MaxPool2d(kernel_size = 3, stride = 1)
        self.pool2 = nn.MaxPool2d(kernel_size = 3, stride = 1)
        
        self.avgpool = nn.AvgPool2d(kernel_size = 27)
        
        # num_layers = 1
        # input size must be # features that come out of conv3
        # since conv3 output = [10, 256, 27, 27] the LSTM input must be 256x27x27 = 186624
        # since do avg. pooling over all: output is of size [10, 256] -> LSTM input = 256
        self.lstm = nn.LSTM(input_size = 256, hidden_size = 128, batch_first = True)
        
        self.fc1 = nn.Linear(128, labels)
        
        self.relu = nn.ReLU()
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        
        CNN_sequence = []
        
        print(x.shape) #batch size, c, t, h, w
        
        for t in range(x.size(2)):
            print('Loop', t)
            out = self.relu(self.conv1(x[:, :, t, :, :]))
            out = self.pool1(out)
        
            out = self.relu(self.conv2(out))
            out = self.pool2(out)
            print('c2: out:', out.shape)
        
            out = self.relu(self.conv3(out))            
            print('c3: out:', out.shape)   # [10, 256, 27, 27]
            
            out.view(out.size(0), -1)
            print('out shape after view:', out.shape) # [10, 256, 27, 27]
            
            CNN_sequence.append(out)
            
        CNN_sequence = torch.stack(CNN_sequence, dim = 0)
        #print('CNN_seq stack:', CNN_sequence.shape)  # 6, 10, 256, 27, 27

        #batch 1st
        CNN_sequence = CNN_sequence.transpose_(0, 1)
        print('CNN_seq going into LSTM:', CNN_sequence.shape)   #10, 6, 256, 27, 27
        
        # do pool before LSTM: avg., max to get info of all layers into shape (B, T, Fetaures)
        # create tensor of shape B, C, H, W
        avg = torch.Tensor(CNN_sequence.size(0), CNN_sequence.size(2), CNN_sequence.size(3), CNN_sequence.size(4))
        
        avg = self.avgpool(avg)
        #print('avg pool:', avg.shape) #[10, 256, 1, 1]
        
        avg = torch.squeeze(avg)
        #print('avg squeeze:', avg.shape) [10, 256]
        
        # only need 10, 6, 256 as input to LSTM (batch size, seuqence length, features)
        CNN_sequence = torch.Tensor(CNN_sequence.size(0), CNN_sequence.size(1), avg.size(1)).to(self.device)
        print('new seq.:', CNN_sequence.shape)
        
        #ready to pitch to LSTM
        self.lstm.flatten_parameters()

        # with -1 for 4 & 5 element only pick the features of the last 25 x25 matrix
        #out, (h_n, c_n) = self.lstm(CNN_sequence[:, :, :, -1, -1], None)
        out, (h_n, c_n) = self.lstm(CNN_sequence[:, :, :], None)
        print('LSTM out:', out.shape)  #batch, seq, feature
        
        # last seq element
        out = self.fc1(out[:, -1, :])
        print('out fc: ', out.shape)
        print(out)
        
        probs = self.sigmoid(out)
        print('probs after sigmoid:', probs)
        
        return out, probs