from torch import nn
import torch

from models.transformer.config import ModelConfig


class RotaryPositionEmbeddings(nn.Module):
    """
    Please note that the following implementation of RoPE differs from that proposed in
    the original RoPE paper (and that contained in the offical Meta Llama release). 
    In particular, _compute_theta() and _rotate() output the same tensor, but with a 
    different permutation of the last axis. The reason for this is that the Hugging Face
    weights (used in this code) were computed by making use of the modified 
    implementation (the same used in this code) and are not compatible with the original 
    implementation. For more details see
    https://github.com/huggingface/transformers/issues/25199 and
    https://discuss.huggingface.co/t/is-llama-rotary-embedding-implementation-correct/44509/3.
    
    The two functions below are those compatible with the original RoPE implementation (and
    the one used in the official Meta Llama release).
    
    def _compute_theta(self, C: int) -> torch.Tensor:
        return torch.tensor(
            [ self.base**((-2*(i-1))/C) for i in range(1, int(C/2)+1) ]
        ).repeat_interleave(2).view(1, 1, 1, C) # (1, 1, 1, C)
        
    @torch.no_grad()  
    def _rotate(self, x: torch.Tensor) -> torch.Tensor:
        ''' 
        rotate a 4D tensor x = (B, H, T, D) on the last axis, so that 
        [x1, x2, x3, x4, ...] becomes [-x2, x1, -x4, x3, ...]
        '''
        x_out = torch.empty_like(x)
        x_out[:, :, :, 0::2] = -x[:, :, :, 1::2]
        x_out[:, :, :, 1::2] = x[:, :, :, 0::2]
        
        return x_out
    """

    
    @torch.no_grad
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.base = config.rope_theta


    @torch.no_grad()
    def _compute_theta(self, C: int) -> torch.Tensor:
        theta = torch.tensor(
            [ self.base**((-2*(i-1))/C) for i in range(1, int(C/2)+1) ]
        )
        return torch.cat((theta, theta), dim=-1)


    @torch.no_grad()  
    def _rotate(self, x: torch.Tensor) -> torch.Tensor:
        ''' 
        rotate a 4D tensor x = (B, H, T, D) on the last axis, so that 
        [x1, x2, x3, x4, ...] becomes [-x_n, -x_n-1, -x_n-3, ..., x3, x2, x1]
        '''
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        
        return torch.cat((-x2, x1), dim=-1)


    @torch.no_grad()
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        B, H, T, D = q.shape
        m = torch.arange(0, T).view(1, 1, T, 1)
        
        theta = self._compute_theta(D) # (1, 1, 1, D)
        
        q = (q * torch.cos(m*theta)) + \
            (self._rotate(q) * torch.sin(m*theta)) # (B, H, T, D)
        k = (k * torch.cos(m*theta)) + \
            (self._rotate(k) * torch.sin(m*theta) ) # (B, H, T, D)
  
        return q, k