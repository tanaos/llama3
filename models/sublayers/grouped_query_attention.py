import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, einsum
import math

from models.transformer.config import ModelConfig
from models.sublayers.rotary_position_embeddings import RotaryPositionEmbeddings


class GroupedQueryAttention(nn.Module):

    def __init__(self, config: ModelConfig, masked: bool =True):
        super().__init__()
        self.masked = masked
        self.num_q_heads = config.num_q_heads
        self.num_kv_heads = config.num_kv_heads
        self.rope = RotaryPositionEmbeddings(config)
        self.k_proj = nn.Linear(config.d_model, config.dim_kv, bias=False)
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.dim_kv, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
    
    def _repeat_heads(self, tensor: torch.Tensor, n_rep: int) -> torch.Tensor:
        '''
        given an input tensor (B, H, T, D), repeats the H dimension n_rep-many times,
        concatenating the new dimensions on the H dimension, thus turning the input
        tensor into a new tensor (B, n_rep*H, T, D)
        
        (B, H, T, D) -> (B, cat(H_1, H_2, ..., H_n_rep), T, D)
        '''
        # tensor must be 4 dimensional: (B, H, T, D)
        assert len(tensor.shape) == 4
        # nothing to do
        if n_rep == 1:
            return tensor
        B, H, T, D = tensor.shape
        # create a new dimension
        tensor = tensor.unsqueeze(2) # (B, H, 1, T, D)
        # broadcast H dimension n_rep-many times on the newly created dimension
        tensor = tensor.expand(B, H, n_rep, T, D) # (B, H, n_rep, T, D)
        # merge H dimension with the newly created dimension
        return rearrange(tensor, 'b h n s d -> b (h n) s d') # (B, n_rep*H, T, D)
        
    
    def forward(self, q_embs, k_embs, v_embs):

        B, T, d_model = q_embs.shape
        
        # make sure tril isn't registered as a model parameter (so it doesn't
        # get trained by the optimizer)
        self.register_buffer('tril', torch.tril(torch.ones(T, T)))
        
        # get keys, queries and values
        q = self.q_proj(q_embs) # (B, T, d_model)
        k = self.k_proj(k_embs) # (B, T, dim_kv)
        v = self.v_proj(v_embs) # (B, T, dim_kv)
        
        # split channel dimension into H (head) and D (head dimension) dimensions, then 
        # swap H and T dimensions for efficiency
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_q_heads) # (B, H, T, D)
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.num_kv_heads) # (B, H, T, D)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.num_kv_heads) # (B, H, T, D)
        
        # apply RoPE to queries and keys
        q, k = self.rope(q, k) # (B, H, T, D), (B, H, T, D)
        num_head_groups = self.num_q_heads // self.num_kv_heads
        
        # if num_q_heads > num_kv_heads, repeat k and v heads num_head_groups-many times
        k = self._repeat_heads(k, num_head_groups) # (B, H, T, D)
        v = self._repeat_heads(v, num_head_groups) # (B, H, T, D)

        # raw attention scores
        raw_scores = q @ k.transpose(-2, -1)
        # scaled attention scores
        scaled_scores = raw_scores * k.shape[-1]**-0.5 # (B, H, T, T)
        # if masked==True, apply masking (past tokens cannot communicate with future ones)
        if self.masked:
            scaled_scores = scaled_scores.masked_fill(
                self.tril[:T, :T] == 0, float('-inf')
            ) # (B, H, T, T)
        # weights
        wei = F.softmax(scaled_scores, dim=-1) # (B, H, T, T)
        # weighted values
        out = wei @ v # (B, H, T, D)
        # swap H and T dimensions back to their original position
        out = out.transpose(1, 2).contiguous() # (B, T, H, D)
        # concatenate H and D dimensions back to the original d_model dimension
        out = out.view(B, T, d_model) # (B, T, d_model)
        # linearly project attention output
        out = self.o_proj(out) # (B, T, d_model)

        return out