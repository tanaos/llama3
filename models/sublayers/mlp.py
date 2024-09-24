import torch
from torch import nn

from models.transformer.config import ModelConfig


class SwiGLUMLP(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.dim_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.dim_ff, bias=False)
        self.down_proj = nn.Linear(config.dim_ff, config.d_model, bias=False)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # perform up projection to increase dimensionality from d_model to dim_ff
        up_projection = self.up_proj(x) # (B, T, dim_ff)
        # perform intermediate, gate projection
        gate_projection = self.gate_proj(x)  # (B, T, dim_ff)
        # calculate swish output, where swish(x) = x*sigmoid(x)
        swish_out = gate_projection * torch.sigmoid(gate_projection) # (B, T, dim_ff)
        # calculate SwiGLU output, where SwiGLU(x) = up_proj(x) * swish(x)
        swiglu_out = up_projection * swish_out # (B, T, dim_ff)
        # project output back to original d_model dimensionality
        mlp_out = self.down_proj(swiglu_out) # (B, T, d_model)

        return mlp_out