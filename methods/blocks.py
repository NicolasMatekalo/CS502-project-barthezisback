import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CasualConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, dilation=1, groups=1, bias=True):
        super(CasualConv1d, self).__init__()
        self.dilation = dilation
        padding = dilation * (kernel_size - 1)
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride,
                                padding, dilation, groups, bias)

    def forward(self, input):
        # Takes something of shape (N, in_channels, T),
        # returns (N, out_channels, T)
        out = self.conv1d(input)
        return out[:, :, :-self.dilation]

class DenseBlock(nn.Module):
    def __init__(self, in_channels, dilation, filters, kernel_size=2):
        super(DenseBlock, self).__init__()
        self.casualconv1 = CasualConv1d(in_channels, filters, kernel_size, dilation=dilation)
        self.casualconv2 = CasualConv1d(in_channels, filters, kernel_size, dilation=dilation)

    def forward(self, input):
        # input is dimensions (N, in_channels, T)
        xf = self.casualconv1(input)
        xg = self.casualconv2(input)
        activations = F.tanh(xf) * F.sigmoid(xg) # shape: (N, filters, T)
        return torch.cat((input, activations), dim=1)
        
class TCBlock(nn.Module):
    def __init__(self, in_channels, seq_length, filters):
        super(TCBlock, self).__init__()
        layer_count = math.ceil(math.log2(seq_length))
        blocks = []
        channel_count = in_channels
        for layer in range(layer_count):
            block = DenseBlock(channel_count, filters, dilation=2**layer)
            blocks.append(block)
            channel_count += filters
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        # input is dimensions (N, T, in_channels)
        input = torch.transpose(input, 1, 2)
        input = self.blocks(input)
        return torch.transpose(input, 1, 2)

class AttentionBlock(nn.Module):
    def __init__(self, in_channels, key_size, value_size):
        super(AttentionBlock, self).__init__()
        self.linear_query = nn.Linear(in_channels, key_size)
        self.linear_keys = nn.Linear(in_channels, key_size)
        self.linear_values = nn.Linear(in_channels, value_size)
        self.sqrt_key_size = math.sqrt(key_size)

    def forward(self, minibatch):
        keys = self.key_layer(minibatch)
        queries = self.query_layer(minibatch)
        values = self.value_layer(minibatch)
        logits = torch.bmm(queries, keys.transpose(2,1))
        mask = logits.data.new(logits.size(1), logits.size(2)).fill_(1).byte()
        mask = torch.triu(mask, 1)
        mask = mask.unsqueeze(0).expand_as(logits)
        logits.data.masked_fill_(mask, float('-inf'))
        probs = F.softmax(logits / self.sqrt_k, dim=2)
        read = torch.bmm(probs, values)
        return torch.cat((minibatch, read), dim=2)
