import torch
from torch.nn import nn


class CM(nn.Module):
    def __init__(self, net):
        