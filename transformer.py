import torch.nn as nn
import torch
import numpy as np

#data format (sample, band, time, height, width)
class CNNVIT(nn.Module):
   def __init__(self, bands=3, labels=24, time =6, device=None):
      super(CNNVIT, self).__init__()

      #channels
      self.ch1, self.ch2, self.ch3, self.ch4 = 32, 64, 128, 256
      #network params
      # height x width 
      self.k1, self.k2, self.k3, self.k4 = (3, 3), (3, 3), (3, 3), (3, 3)
      self.s1, self.s2, self.s3, self.s4 = (1, 1), (1, 1), (1, 1), (1, 1)
      self.p1, self.p2, self.p3, self.p4 = (1, 1), (1, 1), (1, 1), (1, 1)
      self.d1, self.d2, self.d3, self.d4 = (1, 1), (1, 1), (1, 1), (1, 1)


      # network architecture
      # create t CNN models

      self.conv1 = nn.Sequential(
         nn.Conv2d(in_channels=bands, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1, padding=self.p1, dilation=self.d1),
         nn.BatchNorm2d(self.ch1, momentum=0.01),
         nn.ReLU(inplace=True),
         nn.Conv2d(in_channels=self.ch1, out_channels=self.ch1, kernel_size=1, stride=1),
         nn.MaxPool2d(kernel_size=2),
      )
      """
      self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2, padding=self.p2, dilation=self.d2),
            nn.BatchNorm2d(self.ch2, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.ch2, out_channels=self.ch2, kernel_size=1, stride=1),
            nn.MaxPool2d(kernel_size=2),
        )
      self.conv3 = nn.Sequential(
         nn.Conv2d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=self.k3, stride=self.s3, padding=self.p3, dilation=self.d3),
         nn.BatchNorm2d(self.ch3, momentum=0.01),
         nn.ReLU(inplace=True),
         nn.Conv2d(in_channels=self.ch3, out_channels=self.ch3, kernel_size=1, stride=1),
         nn.MaxPool2d(kernel_size=2),
      )
      self.conv4 = nn.Sequential(
         nn.Conv2d(in_channels=self.ch3, out_channels=self.ch4, kernel_size=self.k4, stride=self.s4, padding=self.p4, dilation=self.d4),
         nn.BatchNorm2d(self.ch4, momentum=0.01),
         nn.ReLU(inplace=True),
         nn.Conv2d(in_channels=self.ch4, out_channels=self.ch4, kernel_size=1, stride=1),
         nn.AdaptiveAvgPool2d((1,1)),
      )
      """
      self.global_max_pool = nn.MaxPool2d(kernel_size=112)

      # define single transformer encoder layer
      # self-attention + feedforward network from "Attention is All You Need" paper
      # 4 multi-head self-attention layers each with 64-->512--->64 feedforward network
      transformer_layer = nn.TransformerEncoderLayer(
         d_model=self.ch1, # input feature (frequency) dim after maxpooling 128*563 -> 64*140 (freq*time)
         nhead=4, # 4 self-attention layers in each multi-head self-attention layer in each encoder block
         dim_feedforward=512, # 2 linear layers in each encoder block's feedforward network: dim 64-->512--->64
         dropout=0.4, 
         activation='relu' # ReLU: avoid saturation/tame gradient/reduce compute time
      )
      
      # I'm using 4 instead of the 6 identical stacked encoder layrs used in Attention is All You Need paper
      # Complete transformer block contains 4 full transformer encoder layers (each w/ multihead self-attention+feedforward)
      self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=4)

      #data format (sample x band x height x width)
      #Transformer

      ################# FINAL LINEAR BLOCK ####################
      # Linear softmax layer to take final concatenated embedding tensor 
      #    from parallel 2D convolutional and transformer blocks, output 8 logits 
      # Each full convolution block outputs (64*1*8) embedding flattened to dim 512 1D array 
      # Full transformer block outputs 40*70 feature map, which we time-avg to dim 40 1D array
      # 512*2+40 == 1064 input features --> 8 output emotions 
      self.fc1_linear = nn.Linear(self.ch1,labels) 
      
      ### Softmax layer for the 8 output logits from final FC linear layer 
      self.sigmoid = nn.Sigmoid()
        

   def forward(self, x):
      #define how data passes through model 
      #input shape : data format (sample, band, time, height, width)
      print("WIP ")
      time = x.shape[2]

      cnn_seq = []
      for t in range(time):
         print(x[:, :, t, :, :].size())
         out = self.conv1(x[:, :, t, :, :])
         out = self.global_max_pool(out)
         print(out.size())
         cnn_seq.append(out)

      cnn_seq = torch.stack(cnn_seq, dim=0)
      cnn_seq = torch.squeeze(cnn_seq)

      transformer_output = self.transformer_encoder(cnn_seq)
      time_avg = torch.mean(transformer_output, dim = 0)
      print(time_avg.size())
      
      logits = self.fc1_linear(time_avg) 
      probs = self.sigmoid(logits)
      return logits, probs



      #data format (sample x band x height x width)
