import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import struct
from tqdm import tqdm

# Paths
DDIR = "/kaggle/input/tinypy-language-model/datahack/data/generic-3-digits/"
CHECKPOINT_DIR = "checkpoints/"
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best-model.pth")
deviceid = 0

device = f'cuda:{deviceid}'
print(f"Device set to {device}.")

def log(s: str):
    print(s)

# Ensure model path exists
assert os.path.exists(MODEL_PATH), f"File not found: {MODEL_PATH}. Ensure training completed successfully."

# Load vocabulary size
log("Loading vocab_size")
with open(os.path.join(DDIR, 'vocab_size.txt')) as f:
    vocab_size = int(f.read())

# Model Hyperparameters
block_size = 256
n_embd = 384
n_head = 8
n_layer = 8
dropout = 0

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        scores = (q @ k.transpose(-2, -1)) / (C ** 0.5)
        scores = F.softmax(scores, dim=-1)
        return scores @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=device).unsqueeze(0)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Load Model
log("Loading model")
model = GPT()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

def evaluate():
    log("Evaluating model")
    test_data_path = os.path.join(DDIR, "test.txt")
    with open(test_data_path, "r") as f:
        test_data = f.read()
    examples = test_data.split("\n\n")[:-1]
    
    correct = 0
    for i, example in enumerate(tqdm(examples)):
        input_text = example.split("#STEP\n")[0]
        target_text = example.split("#STEP\n")[1]
        input_ids = torch.tensor([ord(c) for c in input_text], dtype=torch.long).unsqueeze(0).to(device)
        target_ids = torch.tensor([ord(c) for c in target_text], dtype=torch.long).unsqueeze(0).to(device)
        generated_ids = model.generate(input_ids, max_new_tokens=target_ids.shape[1])
        if (generated_ids == target_ids).all():
            correct += 1
    accuracy = correct / len(examples)
    log(f"Evaluation completed. Accuracy: {accuracy * 100:.2f}%")

evaluate()
