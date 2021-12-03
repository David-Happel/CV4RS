import torch.nn as nn
import numpy as np
import helper as h


class C3D(nn.Module):
    """
    The C3D network as described in [1].
    """

    def __init__(self, bands=3, labels=19):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(bands, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2))

        self.conv3a = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2))

        self.conv4a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2))

        self.conv5a = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))

       # d, h, w = h.output_size()
        #print()

        self.fc6 = nn.Linear(8192*8, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, labels)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        h = self.relu(self.conv1(x))
        h = self.pool1(h)
        print(np.shape(h))

        h = self.relu(self.conv2(h))
        h = self.pool2(h)
        print(np.shape(h))

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)
        print(np.shape(h))

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        print("layer 4 conv")
        print(np.shape(h))
        h = self.pool4(h)
        print("layer 4 pool")
        print(np.shape(h))

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        print("layer 5 conv")
        print(np.shape(h))
        h = self.pool5(h)
        print("layer 5 pool")
        print(np.shape(h))

        h = h.view(-1, 8192*8)

        h = self.relu(self.fc6(h))
        h = self.dropout(h)
        h = self.relu(self.fc7(h))
        h = self.dropout(h)
        print(np.shape(h))

        logits = self.fc8(h) 
        print(np.shape(h))

        #probs = self.softmax(logits)

        return logits

"""
References
----------
[1] Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks." 
Proceedings of the IEEE international conference on computer vision. 2015.
"""