import torch.nn as nn
import torch
import numpy as np
import helper as h
import math

#data format (sample, band, time, height, width)
class CNNVIT(nn.Module):

   def __init__(self, bands=3, labels=6, time =6, device=None, d_model = 128, encoder_layers = 1):
      """ Initialise model

      Args:
          bands (int, optional): Number of image bands. Defaults to 3.
          labels (int, optional): Number of class labels. Defaults to 6.
          time (int, optional): Number of timepoints. Defaults to 6.
          device ([type], optional): [description]. Defaults to None.
          d_model (int, optional): Number of features for the encoder. Defaults to 128.
          encoder_layers (int, optional): Number of encoder layers. Defaults to 1.
      """
      super(CNNVIT, self).__init__()

      #NETWORK PARAMETERS 
      #Channels
      self.ch1, self.ch2, self.ch3 = 64, 128, 256
      self.chs = [64, 128, 256]

      #Kernel sizes
      self.k1, self.k2, self.k3 = (3, 3), (3, 3), (3, 3)
      self.k = [(3, 3), (3, 3), (3, 3)]
      # Stride size 
      self.s1, self.s2, self.s3 = (1, 1), (1, 1), (1, 1)
      self.s = [(2, 2), (1, 1), (1, 1)]
      # Padding 
      self.p1, self.p2, self.p3= (1, 1), (1, 1), (1, 1)
      self.p = [(1, 1), (1, 1), (1, 1)]

      self.encoder_layers = encoder_layers

      #ARCHITECHTURE

      self.conv1 = nn.Sequential(
         nn.Conv2d(in_channels=bands, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1, padding=self.p1),
         nn.BatchNorm2d(self.ch1),
         nn.ReLU(inplace=True),
         nn.MaxPool2d(kernel_size=3, stride = 2),
      ) 

      self.conv2 = nn.Sequential(
         nn.Conv2d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2, padding=self.p2),
         nn.BatchNorm2d(self.ch2),
         nn.ReLU(inplace=True),
         nn.MaxPool2d(kernel_size=3, stride = 2),
        )
      
      self.conv3 = nn.Sequential(
         nn.Conv2d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=self.k3, stride=self.s3, padding=self.p3),
         nn.BatchNorm2d(self.ch3),
         nn.ReLU(inplace=True),
         nn.MaxPool2d(kernel_size=3, stride = 2),
      )

      # Getting rid of spatial dimensions by pooling the activation map
      self.global_max_pool = nn.MaxPool2d(kernel_size=27)

      #TRANSFORMER 
      #positional encoder
      self.pos_encoder = PositionalEncoding(self.chs[-1], dropout = 0)

      # define single transformer encoder layer
      # self-attention + feedforward network from "Attention is All You Need" paper
      # multi-head self-attention layers each with self.ch3-->512--->self.ch3 feedforward network
      transformer_layer = nn.TransformerEncoderLayer(
         d_model=self.ch3, # input feature (frequency) dim after maxpooling 
         nhead=8, # 8 self-attention layers in each multi-head self-attention layer in each encoder block
         dim_feedforward=512, # 2 linear layers in each encoder block's feedforward network: dim 64-->512--->64
         dropout=0.1, 
         activation='relu' # ReLU: avoid saturation/tame gradient/reduce compute time
      )
      
      # Complete transformer block contains identical N full transformer encoder layers (each w/ multihead self-attention+feedforward)
      self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=self.encoder_layers)

      #data format (sample x band x height x width)
      #Transformer

      ################# FINAL LINEAR BLOCK ####################
      # Linear softmax layer to take final concatenated embedding tensor 
      #    from parallel 2D convolutional and transformer blocks, output 8 logits 
      # Each full convolution block outputs (64*1*8) embedding flattened to dim 512 1D array 
      # Full transformer block outputs 40*70 feature map, which we time-avg to dim 40 1D array
      # 512*2+40 == 1064 input features --> 8 output emotions 
      self.fc1_linear = nn.Linear(self.ch3,self.ch3) 
      self.fc2_linear = nn.Linear(self.ch3,labels) 
      ### Softmax layer for the 8 output logits from final FC linear layer 
      self.sigmoid = nn.Sigmoid()
        

   def forward(self, x):
      """ forward pass through the network

      Args:
          x (Tensor): input tensor

      Returns:
          Tensor: Tensor with class predictions
      """
      #define how data passes through model 
      #input shape : data format (sample, band, time, height, width)
      # print(f"batch: {x.shape}")
      #CONVOLUTIONAL MODULE
      time = x.shape[2]
      cnn_seq = []
      #convolutional network trained for each timestep
      for t in range(time):
         #convolutions
         out = self.conv1(x[:, :, t, :, :])
         out = self.conv2(out)
         out = self.conv3(out)
         out = self.global_max_pool(out)
         cnn_seq.append(out)

      #stack cnn outputs 
      cnn_seq = torch.stack(cnn_seq, dim=0)
      #reduce dimensionaility
      out = torch.squeeze(cnn_seq)

      #TRANSFORMER MODULE
      #positional encoder 
      out = self.pos_encoder(out)
      #input - timepoints, samples, features
      out = self.transformer_encoder(out)
      #get time averaged features
      time_avg = torch.mean(out, dim = 0)
      
      #LINEAR MODULE
      out = self.fc1_linear(time_avg) 
      logits = self.fc2_linear(out)
      probs = self.sigmoid(logits)
      return logits, probs


class PositionalEncoding(nn.Module):

   def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
      """Initialise positional encoder [1] taken from Pytorch Transformer Tutorial

      Args:
         d_model (int): model dimension (# features at each timepoint)
         dropout (float, optional): ammount of dropout in encoder. Defaults to 0.1.
         max_len (int, optional): max length of input sequence. Defaults to 5000.
      """
         
      super().__init__()
      self.dropout = nn.Dropout(p=dropout)

      position = torch.arange(max_len).unsqueeze(1)
      div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
      pe = torch.zeros(max_len, 1, d_model)
      pe[:, 0, 0::2] = torch.sin(position * div_term)
      pe[:, 0, 1::2] = torch.cos(position * div_term)
      self.register_buffer('pe', pe)

   def forward(self, x):
      """ Add positional encoding to input sequence
      Args:
         x: Tensor, shape [seq_len, batch_size, embedding_dim]
      """
      x = x + self.pe[:x.size(0)]
      return self.dropout(x)

"""
References
----------
[1] https://pytorch.org/tutorials/beginner/transformer_tutorial.html
"""