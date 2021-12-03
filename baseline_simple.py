import torch.nn as nn
import torch as t
import numpy as np


class C3D(nn.Module):
    """
    The C3D network as described in [1].
    """

    def __init__(self, bands=3, labels=19):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(bands, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(1, 2, 2))
        
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2))

        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2))

        self.conv4 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2))

        self.conv5 = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5= nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2))

        self.fc6 = nn.Linear(12544, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, labels)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)
       
        h = self.relu(self.conv3(h))
        h = self.pool3(h)

        h = self.relu(self.conv4(h))
        h = self.pool4(h)

        h = self.relu(self.conv5(h))
        h = self.pool5(h)

        h = t.flatten(h, start_dim = 1)

        h = self.relu(self.fc6(h))
        h = self.dropout(h)
        h = self.relu(self.fc7(h))
        h = self.dropout(h)

        logits = self.fc8(h) 
      
        return logits


"""
References
----------
[1] Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks." 
Proceedings of the IEEE international conference on computer vision. 2015.
"""