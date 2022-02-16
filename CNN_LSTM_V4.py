import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as f

class CNN_LSTM(nn.Module):

    def __init__(self, bands = 3, labels = 6, time=6, lstm_layers = 1):
        """[summary]

        Args:
            bands (int, optional): [description]. Defaults to 3.
            labels (int, optional): [description]. Defaults to 6.
            time (int, optional): [description]. Defaults to 6.
            lstm_layers (int, optional): [description]. Defaults to 1.
        """
        super(CNN_LSTM, self).__init__()
        
        # channels
        self.ch1, self.ch2, self.ch3 = 64, 128, 256
        
        # args. where H = W
        self.kconv, self.kpool, self.kavg = 3, 3, 27
        self.sconv, self.spool = 1, 2
        self.p = 1
        self.lstm_layers = lstm_layers

        if lstm_layers == 1:
            self.lstm_h = 128
            self.dropout = 0.0
        if lstm_layers == 2:
            self.lstm_h = 128
            self.dropout = 0.1
        if lstm_layers == 3:
            self.lstm_h = 128
            self.dropout = 0.1

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = bands, out_channels = self.ch1, kernel_size = self.kconv, stride = self.sconv, padding = self.p),
            nn.BatchNorm2d(num_features = self.ch1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = self.kpool, stride = self.spool)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = self.ch1, out_channels = self.ch2, kernel_size = self.kconv, stride = self.sconv, padding = self.p),
            nn.BatchNorm2d(num_features = self.ch2),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = self.kpool, stride = self.spool)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels = self.ch2, out_channels = self.ch3, kernel_size = self.kconv, stride = self.sconv, padding = self.p),
            nn.BatchNorm2d(num_features = self.ch3),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = self.kpool, stride = self.spool)
        )

        self.global_max_pool = nn.MaxPool2d(kernel_size = self.kavg)
        
        self.lstm = nn.LSTM(
            input_size = self.ch3,
            hidden_size = self.lstm_h,
            num_layers = self.lstm_layers,
            batch_first = True,
            dropout = self.dropout
        )
        
        self.fc = nn.Linear(
            in_features = self.lstm_h,
            out_features = labels
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x): #x = [B, C, T, H, W]
        """[summary]

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        
        CNN_sequence = [] 
        
        for t in range(x.size(2)): 

            out = self.conv1(x[:, :, t, :, :])
            #print('c1: out:', out.shape) #[10, 64, 111, 111]
        
            out = self.conv2(out)
            #print('c2: out:', out.shape) #[10, 128, 55, 55]
        
            out = self.conv3(out)          
            #print('c3: out:', out.shape) #[10, 256, 27, 27]

            out = self.global_max_pool(out)
            #print('avg. pool:', out.shape) #[10, 256, 1, 1]
            
            CNN_sequence.append(out)
        
        # stack over time dim.
        CNN_sequence = torch.stack(CNN_sequence, dim = 0)
        #print('CNN_seq stack:', CNN_sequence.shape)
        
        # squeeze to remove 1 dims.
        CNN_sequence = torch.squeeze(CNN_sequence)
        #print('CNN_seq squeeze:', CNN_sequence.shape)

        # transpose to get batch 1st
        CNN_sequence = CNN_sequence.transpose_(0, 1)
        #print('CNN_seq going into LSTM:', CNN_sequence.shape)
        
        # makes computation easier
        self.lstm.flatten_parameters()
        
        # pitching to LSTM
        out, (h_n, c_n) = self.lstm(CNN_sequence[:, :, :], None)
        #print('LSTM out:', out.shape)
        
        # take output of last sequence element only
        out = self.fc(out[:, -1, :])
        #print('out fc: ', out.shape)
        #print(out)
        
        # return probs
        probs = self.sigmoid(out)
        #print('probs after sigmoid:', probs)
        
        return out, probs