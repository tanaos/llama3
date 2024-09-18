from torch import nn
import torch


class RotaryPositionEmbeddings(nn.Module):
    
    @torch.no_grad
    def __init__(self, config):
        super().__init__()
        self.base = config.rope_theta
    
    @torch.no_grad()
    def _compute_theta(self, C):
        # TODO: check why the commented out part does not work
        #return torch.tensor(
        #    [ self.base**((-2*(i-1))/C) for i in range(1, int(C/2)+1) ]
        #).repeat_interleave(2).view(1, 1, 1, C) # (1, 1, 1, C)
        
        theta = torch.tensor(
            [ self.base**((-2*(i-1))/C) for i in range(1, int(C/2)+1) ]
        )
        return torch.cat((theta, theta), dim=-1)

      
    @torch.no_grad()  
    def _rotate(self, x):
        ''' 
        rotate a 4D tensor x = (B, H, T, D) on the last axis, so that 
        [x1, x2, x3, x4, ...] becomes [-x_n, -x_n-1, -x_n-3, ..., x3, x2, x1]
        '''
        # TODO: check why the commented out part does not work
        #x_out = torch.empty_like(x)
        #x_out[:, :, :, 0::2] = -x[:, :, :, 1::2]
        #x_out[:, :, :, 1::2] = x[:, :, :, 0::2]
        #
        #return x_out
        
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
        
        
    @torch.no_grad()
    def forward(self, q, k):
        B, H, T, D = q.shape
        m = torch.arange(0, T).view(1, 1, T, 1)
        
        theta = self._compute_theta(D) # (1, 1, 1, D)
        
        q = (q * torch.cos(m*theta)) + \
            (self._rotate(q) * torch.sin(m*theta)) # (B, H, T, D)
        k = (k * torch.cos(m*theta)) + \
            (self._rotate(k) * torch.sin(m*theta) ) # (B, H, T, D)
  
        return q, k