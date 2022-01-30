import torch.nn as nn
import torch as t
import numpy as np
import torch.nn.functional as f


class C3D(nn.Module):
    """
    The C3D network inpired by [1].
    ----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
    ================================================================
            Conv3d-1        [-1, 32, 6, 224, 224]       2,624
            ReLU-2          [-1, 32, 6, 224, 224]       0
            MaxPool3d-3     [-1, 32, 5, 112, 112]       0
            Conv3d-4        [-1, 64, 5, 112, 112]       55,360
            ReLU-5          [-1, 64, 5, 112, 112]       0
            MaxPool3d-6     [-1, 64, 4, 56, 56]         0
            Conv3d-7        [-1, 128, 4, 56, 56]        221,312
            ReLU-8          [-1, 128, 4, 56, 56]        0
            MaxPool3d-9     [-1, 128, 3, 28, 28]        0
            Conv3d-10       [-1, 256, 3, 28, 28]        884,992
            ReLU-11         [-1, 256, 3, 28, 28]        0
            MaxPool3d-12    [-1, 256, 2, 14, 14]        0
            Conv3d-13       [-1, 256, 2, 14, 14]        1,769,728
            ReLU-14         [-1, 256, 2, 14, 14]        0
            MaxPool3d-15    [-1, 256, 1, 7, 7]          0
            Linear-16       [-1, 4096]                  51,384,320
            ReLU-17         [-1, 4096]                  0
            Dropout-18      [-1, 4096]                  0
            Linear-19       [-1, 4096]                  16,781,312
            ReLU-20         [-1, 4096]                  0
            Dropout-21      [-1, 4096]                  0
            Linear-22       [-1, 18]                    73,746
    ================================================================
    """

    def __init__(self, bands=3, labels=24, time=6, device="cpu"):
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

        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        h = f.relu(self.conv1(x))
        h = self.pool1(h)

        h = f.relu(self.conv2(h))
        h = self.pool2(h)
       
        h = f.relu(self.conv3(h))
        h = self.pool3(h)

        h = f.relu(self.conv4(h))
        h = self.pool4(h)

        h = f.relu(self.conv5(h))
        h = self.pool5(h)

        h = t.flatten(h, start_dim = 1)

        h = f.relu(self.fc6(h))
        h = self.dropout1(h)
        h = f.relu(self.fc7(h))
        h = self.dropout2(h)

        logits = self.fc8(h) 
        probs = self.sigmoid(logits)
      
        return logits, probs


"""
References
----------
[1] Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks." 
Proceedings of the IEEE international conference on computer vision. 2015.
"""