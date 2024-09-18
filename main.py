import torch
from transformers import AutoTokenizer

from models.transformer.transformer import Transformer


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

initial_sentence = "Hello, I'm a language model,"
initial_tokens = torch.tensor([tokenizer.encode(initial_sentence)])

model = Transformer.from_pretrained()
'''
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = CaesarConfig(
    B = 4,
    T = 32,
    d_model = 4096,
    dim_kv = 1024,
    num_q_heads = 32,
    num_kv_heads = 8,
    dim_ff = 14336,
    vocab_size = 128256,
    n_layer = 1
)
model = Transformer(device, config)
'''

out_tokens = model.generate(idx=initial_tokens, max_new_tokens=20)
print(tokenizer.batch_decode(out_tokens))