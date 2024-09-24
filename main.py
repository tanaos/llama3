import torch
from transformers import AutoTokenizer
from huggingface_hub import login
import os
from dotenv import load_dotenv

from models.transformer.transformer import Transformer


load_dotenv()
login(token=os.getenv("HF_TOKEN"))
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

initial_sentence = "Hello, I'm a language model,"
initial_tokens = torch.tensor([tokenizer.encode(initial_sentence)])

model = Transformer.from_pretrained()
out_tokens = model.generate(idx=initial_tokens, max_new_tokens=20)
print(tokenizer.batch_decode(out_tokens))