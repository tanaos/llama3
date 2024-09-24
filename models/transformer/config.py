from dataclasses import dataclass


# the default values are those used in the Hugging Face implementation of LLAMA 3
@dataclass
class ModelConfig:
    d_model: int = 4096
    dim_kv: int = 1024
    num_q_heads: int = 32
    num_kv_heads: int = 8
    dim_ff: int = 14336
    vocab_size: int = 128256
    n_layer: int = 32
    norm_eps: int = 1e-05
    rope_theta: float = 500000.0