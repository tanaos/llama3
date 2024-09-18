import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(config.d_model))
        self.eps = config.norm_eps

    def forward(self, x):
        return (x*torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)) * self.weight