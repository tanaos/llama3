from torch import nn
import torch

from models.sublayers.mlp import SwiGLUMLP
from models.sublayers.grouped_query_attention import GroupedQueryAttention
from models.normalization.rms_norm import RMSNorm


class Decoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config)
        self.post_attention_layernorm = RMSNorm(config)
        self.mlp = SwiGLUMLP(config)
        self.self_attn = GroupedQueryAttention(config)


    def forward(self, output_embs):
        # 1. normalize input with Root Mean Square (RMS) layer norm
        norm_input = self.input_layernorm(output_embs)
        # 2. compute self-attention output via Masked Multi-Head self-attention
        self_att_output = self.self_attn(
            q_embs=norm_input, k_embs=norm_input, v_embs=norm_input # (B, T, d_model)
        )
        # 3. apply residual connection
        self_att_output += output_embs # (B, T, d_model)
        # 4. apply RMS layer norm to self-attention output
        norm_self_att_output = self.post_attention_layernorm(self_att_output) # (B, T, d_model)
        # 5. pass normalized self-attention output through a SwiGLUMLP
        mlp_output = self.mlp(norm_self_att_output) # (B, T, d_model)
        # 6. apply residual connection to mlp output
        mlp_output += self_att_output # (B, T, d_model)

        return mlp_output