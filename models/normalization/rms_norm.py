import torch
from torch import nn

from models.transformer.config import ModelConfig


class RMSNorm(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(config.d_model))
        self.eps = config.norm_eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x*torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)) * self.weight