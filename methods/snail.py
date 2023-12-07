import math
import torch
import torch.nn as nn

from blocks import *
from methods.meta_template import MetaTemplate

class SNAIL(MetaTemplate):
    def __init__(self, backbone, n_way, n_support):
        # N-way, K-shot
        super(SNAIL, self).__init__(backbone, n_way, n_support, change_way=True)
        
        num_channels = 64 + n_way # CHANGE 64 TO CORRECT DIM
        num_filters = int(math.ceil(math.log(n_way * n_support + 1, 2)))
        self.attention1 = AttentionBlock(num_channels, 64, 32)
        
        num_channels += 32
        self.tc1 = TCBlock(num_channels, n_way * n_support + 1, 128)
        num_channels += num_filters * 128
        self.attention2 = AttentionBlock(num_channels, 256, 128)
        
        num_channels += 128
        self.tc2 = TCBlock(num_channels, n_way * n_support + 1, 128)
        num_channels += num_filters * 128
        self.attention3 = AttentionBlock(num_channels, 512, 256)
        
        num_channels += 256
        self.fc = nn.Linear(num_channels, n_way)

    def forward(self, input, labels):
        x = self.encoder(input)
        batch_size = int(labels.size()[0] / (self.n_way * self.n_support + 1))
        last_idxs = [(i + 1) * (self.n_way * self.n_support + 1) - 1 for i in range(batch_size)]
        labels[last_idxs] = torch.Tensor(np.zeros((batch_size, labels.size()[1]))).to(self.device)
        
        x = torch.cat((x, labels), 1)
        x = x.view((batch_size, self.n_way * self.n_support + 1, -1))
        x = self.attention1(x)
        x = self.tc1(x)
        x = self.attention2(x)
        x = self.tc2(x)
        x = self.attention3(x)
        x = self.fc(x)
        
        return x