import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
import os
from dotenv import load_dotenv

from models.blocks.decoder import Decoder
from models.normalization.rms_norm import RMSNorm


# the default values are those used in the Hugging Face implementation of LLAMA 3
@dataclass
class CaesarConfig:
    d_model: int = 4096
    dim_kv: int = 1024
    num_q_heads: int = 32
    num_kv_heads: int = 8
    dim_ff: int = 14336
    vocab_size: int = 128256
    n_layer: int = 32
    norm_eps: int = 1e-05
    rope_theta: float = 500000.0


class Transformer(nn.Module):

    def __init__(self, device, config):
        super().__init__()
        self.device = device
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([ Decoder(config) for _ in range(config.n_layer) ])
        self.norm = RMSNorm(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)


    def forward(self, dec_input, targets=None):
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
    def generate(self, idx, max_new_tokens):
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
    def from_pretrained(cls):
        print("loading weights from pretrained model")
        
        load_dotenv()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        config = CaesarConfig()
        model = Transformer(device, config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        
        from transformers import AutoModelForCausalLM
        from huggingface_hub import login
        login(token=os.getenv("HF_TOKEN"))
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