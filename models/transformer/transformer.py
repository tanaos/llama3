import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Type, TypeVar, Optional

from models.blocks.decoder import Decoder
from models.normalization.rms_norm import RMSNorm
from models.transformer.config import ModelConfig


T = TypeVar("T", bound="Transformer")


class Transformer(nn.Module):

    def __init__(self, device: str, config: ModelConfig):
        super().__init__()
        self.device = device
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([ Decoder(config) for _ in range(config.n_layer) ])
        self.norm = RMSNorm(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)


    def forward(
        self, dec_input: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[float]]:
        # compute output embeddings
        x = self.embed_tokens(dec_input) # (B, T, d_model)
        # sequentially run output embeddings through all decoder layers
        for layer in self.layers:
            x = layer(x)
        # normalize output of the final layer
        x = self.norm(x)
        # run decoder output through the language model head to get logits
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets == None:
            loss = None
        else:
            B, T, vocab_size = logits.shape
            logits = logits.view(B*T, vocab_size) # (B*T, vocab_size)
            targets = targets.view(B*T) # (B*T)
            probs = F.softmax(logits, dim=1) # (B*T, vocab_size)
            logprobs = torch.log(probs) # (B*T, vocab_size)
            relevant_logprobs = logprobs[torch.arange(B*T), targets] # (B*T)
            #loss = F.cross_entropy(logits, targets)
            loss = -sum(relevant_logprobs)/len(relevant_logprobs)

        return logits, loss


    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        out = idx # (len(idx)), increases at each iteration
        
        for _ in range(max_new_tokens):
            # compute logits with a forward pass
            logits, loss = self(out) # (B, len(out), vocab_size)
            # only focus on the last token"s logits
            logits = logits[:, -1, :] # (B, vocab_size)
            # generate a probability distribution for the next token
            probs = F.softmax(logits, dim=-1) # (B, vocab_size)
            # sample next token from probability distribution
            next_token = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled token to output
            out = torch.cat((out, next_token), dim=1) # (B, len(out) + 1)
            
        return out
    

    @classmethod
    def from_pretrained(cls: Type[T]) -> T:
        print("loading weights from pretrained model")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        config = ModelConfig()
        model = Transformer(device, config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        
        from transformers import AutoModelForCausalLM
        model_hf = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct"
        )
        sd_hf = model_hf.state_dict()
        
        sd_keys_hf = sd_hf.keys()
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            # vanilla copy over the other parameters
            assert sd_hf[k].shape == sd[k.replace("model.", "")].shape
            with torch.no_grad():
                sd[k.replace("model.", "")].copy_(sd_hf[k])

        print("all done")
        
        return model