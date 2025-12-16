import numpy as np
import os
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import alexnet

import config as c
from freia_funcs import permute_layer, glow_coupling_layer, F_fully_connected, ReversibleGraphNet, OutputNode, \
    InputNode, Node

def nf_head(input_dim=c.n_feat):
    nodes = list()
    nodes.append(InputNode(input_dim, name='input'))
    for k in range(c.n_coupling_blocks):
        nodes.append(Node([nodes[-1].out0], permute_layer, {'seed': k}, name=F'permute_{k}'))
        nodes.append(Node([nodes[-1].out0], glow_coupling_layer,
                          {'clamp': c.clamp_alpha, 'F_class': F_fully_connected,
                           'F_args': {'internal_size': c.fc_internal, 'dropout': c.dropout}},
                          name=F'fc_{k}'))
    nodes.append(OutputNode([nodes[-1].out0], name='output'))
    coder = ReversibleGraphNet(nodes)
    return coder


class flow_model(nn.Module):
    def __init__(self):
        super(flow_model, self).__init__()

        self.nf = nf_head(input_dim = 1024)

    def forward(self, x):
        z = self.nf(x)
        return z
    
class flow_model_multi_fc(nn.Module):
    def __init__(self):
        super(flow_model_multi_fc, self).__init__()
        self.fc1 = torch.nn.Linear(1024, 512)
        self.relu = torch.nn.LeakyReLU(0.2)
        self.fc2 = torch.nn.Linear(512, 256)

        self.nf = nf_head(input_dim = 256)

    def forward(self, x):
        res_x = self.fc2(self.relu((self.fc1(x)))) 
        z = self.nf(res_x)
        return z
    
