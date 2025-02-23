import random
import     os
import     time
import     torch
import     datetime
import     math
import  inspect
import     torch.nn             as nn
import     torch.nn.functional as F
import     numpy                 as np
import     torch.distributed     as dist
from     torch.nn.parallel     import DistributedDataParallel as DDP

# Set the device ids
deviceids     = [0, 1]

# Set the data directory
data_dirs = [
    "/kaggle/input/tinypy-language-model/datahack/data/generic-3-digits/",
    "/kaggle/input/tinypy-language-model/datahack/data/arithmetics/",
    "/kaggle/input/tinypy-language-model/datahack/data/conditionals/"
]

# Set arch-hyperparams for the GPT model
block_size     = 256     # Maximum context length
n_embd = 384  # Ensure it's a multiple of n_head
n_head = 8  # Increase heads for better parallelization
n_layer = 8  # Increase layers for better learning


batch_size = 190       # Batch size for training

# Check if this is a ddp run
ddp = int(os.environ.get('RANK', -1)) != -1

# If ddp than setup per process control variables
if ddp:
    print("ddp is set to True")
    from torch.distributed import init_process_group, destroy_process_group
    assert torch.cuda.is_available()
    init_process_group(backend='nccl', timeout=datetime.timedelta(minutes=60))
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    assert ddp_world_size == len(deviceids)
    device = f'cuda:{deviceids[ddp_rank]}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
else:
    print('ddp is set to False')
    assert len(deviceids) == 1
    master_process = True
    ddp_rank = 0
    seed_offset = 0
    ddp_world_size = 1
    device = f'cuda:{deviceids[ddp_rank]}'
    torch.cuda.set_device(device)

if master_process:

    # Prepare the checkpoints folder
    os.makedirs(os.path.dirname("checkpoints/"), exist_ok=True)
    
    # Open the log file
    log_file = open("train.log", "w")

    # Create the log boilerplate for progress bars
    pbar_recept_string = " " * 200 + "\n"
    log_file.write(pbar_recept_string)
    log_file.write(pbar_recept_string)
    log_file.flush()

    # Define the log function
    def log(s:str, p_level=None):
        if p_level == 1:
            log_file.seek(0,0)
            log_file.write(pbar_recept_string)
            log_file.seek(0,0)
            log_file.write(s)
            log_file.seek(0,2)
        elif p_level == 2:
            log_file.seek(len(pbar_recept_string), 0)
            log_file.write(pbar_recept_string)
            log_file.seek(len(pbar_recept_string), 0)
            log_file.write(s)
            log_file.seek(0,2)
        else:
            log_file.write('\n' + s + '\n')
            print(s)
        log_file.flush()

    # Define function to convert seconds to days, hours, minutes, seconds
    def convert_seconds(seconds:float):
        # ignoring the sub seconds 
        seconds = int(seconds)
        days, seconds = divmod(seconds, 86400)
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        return (days, hours, minutes, seconds)

    batch_log = open('batch_log.log', 'w')

# Set the random seed for reproducibility
seed = 42 + seed_offset
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


with open(data_dirs[0] + "vocab_size.txt", "rb") as f:
    vocab_size = int(f.read())
if master_process: log(f"vocab_size: {vocab_size}")

# Load train.bin
if master_process: log("Loading train.bin")
before         = time.time()
train_data     = np.concatenate([np.memmap(d + "train.bin", dtype=np.uint8, mode="r") for d in data_dirs])
train_data     = np.array(train_data)
after         = time.time()
if master_process: log(f"took {convert_seconds(after - before)}")

# Load val.bin
if master_process: log("Loading val.bin")
before         = time.time()
val_data = np.concatenate([np.memmap(d + "val.bin", dtype=np.uint8, mode="r") for d in data_dirs])
val_data     = np.array(val_data)
after         = time.time()
if master_process: log(f"took {convert_seconds(after - before)}")


# Set the train-hyperparams
batch_nb_tokens = batch_size * block_size
assert batch_size % len(deviceids) == 0 # for now just to make sure
microbatch_size = batch_size // len(deviceids) # micro batch size == batch size per device
microbatch_nb_tokens = microbatch_size * block_size # Number of tokens in a micro-batch
acum_steps = 1
assert microbatch_size % acum_steps == 0
nano_batch_size = microbatch_size // acum_steps
dropout = 0 # Dropout rate
max_pseudo_epochs = 1    # Number of pseudo-epochs to train for
max_iters = int( ( max_pseudo_epochs * len(train_data) ) / ( batch_size * block_size ) )
epoch_iters = int(len(train_data) / (batch_size * block_size))
learning_rate = 1e-3     # Initial Learning rate value
decay_lr = True
warmup_iters = int(0.1 * max_iters)
lr_decay_iters = max_iters
assert lr_decay_iters <= max_iters
assert warmup_iters <= lr_decay_iters
min_lr = learning_rate * 0.1
beta1 = 0.9 # Adam beta1
beta2 = 0.95 # Adam beta2
weight_decay = 1e-1 # Weight decay
grad_clip = 1.0 # Gradient clipping value
max_degradations = -1     # number of consecutive degradations on val loss before stoping the training
miles = [int(max_iters * m) for m in [0.7, 0.8, 0.9]]  # Milestones for learning rate decay as fractions of max_iters
cuda_sync = False
compile = False
compile_mode = 'default'


# Set the evaluation-hyperparams
eval_interval = int(epoch_iters * 0.1) # Evaluation interval
eval_iters = 100  # Number of iterations for evaluation

# __Define some utils__

# Define function to get batch of token chunks
def get_batch(split):
    data = train_data if split == 'train' else val_data
    idx = random.randint(0, len(data) - microbatch_nb_tokens - 1)
    
    x = torch.from_numpy((data[idx:idx+microbatch_nb_tokens]).astype(np.int64)).view(microbatch_size, block_size)
    y = torch.from_numpy((data[idx+1:idx+1+microbatch_nb_tokens]).astype(np.int64)).view(microbatch_size, block_size)
    x, y = x.to(device), y.to(device)

    return x, y

# Define the evaluation batch size
eval_batch_size = 128
eval_batch_nb_tokens = eval_batch_size * block_size

# Define the evaluation batch loader
def load_eval_batch(split):
    data = train_data if split == 'train' else val_data
    idx = random.randint(0, len(data) - eval_batch_nb_tokens - 1)
    x = torch.from_numpy((data[idx:idx+eval_batch_nb_tokens]).astype(np.int64)).view(eval_batch_size, block_size)
    y = torch.from_numpy((data[idx+1:idx+1+eval_batch_nb_tokens]).astype(np.int64)).view(eval_batch_size, block_size)
    x, y = x.to(device), y.to(device)
    return x, y

# Define function to estimate loss on train and val splits
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    eval_start_time = time.time()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            past = time.time()
            X, Y = load_eval_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
            present = time.time()
            log(f"{split}>|ITERS: {k+1} / {eval_iters} | COMP: {(k+1)/eval_iters * 100:.2f}% | RATE: {1/(present-past):.2f} it./s | SPD: {present - past :.4f} s/it.| ERT: {convert_seconds((eval_iters-k-1) * (present-past))} | ET: {convert_seconds(time.time()-eval_start_time)}", p_level = 2)
        out[split] = losses.mean()
    model.train()
    return out

# Define function to make large numbers of parameters human-readable
def human_readable(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.0f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# __Define the model__

class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   
        q = self.query(x)
        v = self.value(x)
        
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout if self.training else 0, is_causal=True)
        
        return out
class MemoryAugmentedAttention(nn.Module):
    def __init__(self, num_heads, head_size, memory_size=256):  # Match block_size
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.memory = nn.Parameter(torch.randn(memory_size, num_heads * head_size))  # Ensure correct shape
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # Concatenate across heads

        # Ensure memory shape matches output embedding size
        batch_size, seq_len, emb_dim = out.shape
        if seq_len <= self.memory.shape[0]:
            memory_slice = self.memory[:seq_len, :].unsqueeze(0)  # Ensure matching sequence length
        else:
            memory_slice = self.memory.unsqueeze(0).expand(batch_size, -1, -1)

        out = out + memory_slice
        return self.dropout(self.proj(out))




class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    """Gated Linear Unit (GLU) based feedforward for efficient training."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),  # Replacing ReLU with GELU for better gradient flow
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class ExecutionStepEmbedding(nn.Module):
    """Encodes execution step tracking for better variable tracking."""

    def __init__(self, num_steps, embedding_dim):
        super().__init__()
        self.step_embedding = nn.Embedding(num_steps, embedding_dim)

    def forward(self, step_ids):
        return self.step_embedding(step_ids)
        
class Block(nn.Module):
    """Pre-normalized transformer block with memory-augmented attention."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MemoryAugmentedAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd, bias=False)
        self.ln2 = nn.LayerNorm(n_embd, bias=False)

    def forward(self, x):
        x = x + 0.1 * self.sa(self.ln1(x))  # Scale residual connections
        x = x + 0.1 * self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    """Enhanced GPT model with execution step tracking and memory-augmented attention."""

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.step_embedding = ExecutionStepEmbedding(block_size, n_embd)  # Execution step tracking
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd, bias=False)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None, step_ids=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        step_emb = self.step_embedding(step_ids) if step_ids is not None else torch.zeros_like(tok_emb)


        x = tok_emb + pos_emb + step_emb  # Merge embeddings
        x = self.blocks(x)  # Transformer layers
        x = self.ln_f(x)  # Final normalization

        logits = self.lm_head(x)  # Output layer

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    @torch.no_grad()
    def generate(model, compiled_model, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in tqdm(range(max_new_tokens)):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:] # (B, T)
            # get the predictions
            if idx_cond.shape[1] < block_size: # if we have less than block_size tokens, we need to use the eager model
                logits, loss = model(idx_cond)
            else: # if we have block_size tokens, we can now use the compiled model to speedup inference
                loc_tens = torch.ones((1, block_size), dtype=torch.long, device=device).copy_(idx_cond)
                logits, loss = compiled_model(loc_tens)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# Create the model
model = GPT()
model.to(device)
num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
num_parameters_hr = human_readable(num_parameters)
if master_process:
    log(f'The model has {num_parameters_hr} trainable parameters')
    print(f'The model has {num_parameters_hr} trainable parameters')

# Wrap the model in DDP and/or compilation
train_model = model

# If ddp then wrap the model in DDP
if ddp:
    train_model = DDP(train_model, device_ids=[deviceids[ddp_rank]])

# If compile == True compile the model
if compile:
    if master_process: log("Compiling the model... (takes a ~minute)")
    train_model = torch.compile(train_model, mode=compile_mode)

param_dict = {pn: p for pn, p in train_model.named_parameters()}
# filter out those that do not require grad
param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
# create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
# i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
optim_groups = [
    {'params': decay_params, 'weight_decay': weight_decay},
    {'params': nodecay_params, 'weight_decay': 0.0}
]
num_decay_params = sum(p.numel() for p in decay_params)
num_nodecay_params = sum(p.numel() for p in nodecay_params)
print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
# Create AdamW optimizer and use the fused version if it is available
fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
use_fused = fused_available and ('cuda' in device)
extra_args = dict(fused=True) if use_fused else dict()
optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(beta1, beta2), **extra_args)
print(f"using fused AdamW: {use_fused}")


# Set the torch precision to tf32
torch.set_float32_matmul_precision('high')

train_start_time = time.time()

if master_process:
    log("Evaluating initial loss")
    last_losses = estimate_loss()
    log(f'INITIAL LOSS: train loss {last_losses["train"]} | val loss {last_losses["val"]}')
    best_val_loss = last_losses["val"]

print('Current Device:', torch.cuda.current_device())

# Training Loop
start_iter = 0
nb_time_samples = 0
deltatimes = 0
tl_max_warmup_iters = 20
tl_warmup_iters = 0
for iter in range(start_iter, max_iters):

    # Do one forward backward pass
    past = time.time()
    
    # Clear the gradients
    optimizer.zero_grad(set_to_none=True)
    
    xb, yb = get_batch('train')

    logits, loss = train_model(xb, yb)

    loss.backward()

    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(train_model.parameters(), grad_clip)
    
    micro_loss = loss.detach()

    if ddp:
        dist.all_reduce(micro_loss, op=dist.ReduceOp.AVG)

    lr = get_lr(iter) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    

    optimizer.step()

    if cuda_sync:
        torch.cuda.synchronize(device)

    present = time.time()

    if master_process:
        
        if tl_warmup_iters <= tl_max_warmup_iters:
            log(f"INIT ITER TIME: {present-past}")
            log(f"|BATCH_LOSS: {micro_loss:.5f} | TPS: {int(batch_size*block_size/(present-past)):8} tokens/sec. | ITERS: {iter+1} / {max_iters} | EPOCHS: {(block_size * batch_size * (iter+1))/len(train_data):.2f} | COMP: {(iter+1)/max_iters * 100:.2f}% | LR: {lr:.4f} | SPD: {(present - past) * 1000 :.4f} ms/it.| ERT: {convert_seconds((max_iters-iter-1) * (present-past))} | ET: {convert_seconds(present-train_start_time)}", p_level = 1)
            print(f"|BATCH_LOSS: {micro_loss:.5f}|TPS: {int(batch_size*block_size/(present-past)):8} tokens/sec.|ITERS: {iter+1} / {max_iters}|EPOCHS: {(block_size * batch_size * (iter+1))/len(train_data):.2f}|COMP: {(iter+1)/max_iters * 100:.2f}%|LR: {lr:.4f} it./s|SPD: {(present - past) * 1000 :.4f} ms/it|ERT: {convert_seconds((max_iters-iter-1) * (present-past))}|ET: {convert_seconds(present-train_start_time)}")
            tl_warmup_iters += 1
        else:
            deltatimes += present-past
            nb_time_samples += 1
            mean_deltatime = deltatimes / nb_time_samples # mean deltatime
            log(f"|BATCH_LOSS: {micro_loss:.5f} | TPS: {int(batch_size*block_size/(present-past)):8} tokens/sec. | ITERS: {iter+1} / {max_iters} | EPOCHS: {(block_size * batch_size * (iter+1))/len(train_data):.2f} | COMP: {(iter+1)/max_iters * 100:.2f}% | LR: {lr:.4f} it./s | SPD: {(present - past) * 1000 :.4f} ms/it.| ERT: {convert_seconds((max_iters-iter-1) * mean_deltatime)} | ET: {convert_seconds(present-train_start_time)}", p_level = 1)
            print(f"|BATCH_LOSS: {micro_loss:.5f}|TPS: {int(batch_size*block_size/(present-past)):8} tokens/sec.|ITERS: {iter+1} / {max_iters}|EPOCHS: {(block_size * batch_size * (iter+1))/len(train_data):.2f}|COMP: {(iter+1)/max_iters * 100:.2f}%|LR: {lr:.4f} it./s|SPD: {(present - past) * 1000 :.4f} ms/it|ERT: {convert_seconds((max_iters-iter-1) * mean_deltatime)}|ET: {convert_seconds(present-train_start_time)}")
        
        batch_log.write(f"BATCH_LOSS: {micro_loss:.5f}\n")
        batch_log.flush()
    
    # If we reach the evaluation interval, evaluate the model on train.bin and val.bin and checkpoint it
    if ((iter+1) % eval_interval == 0 or (iter == max_iters - 1)) and master_process:
        
        log("\nEvaluation and checkpointing ...\n")
        epoch = (block_size * batch_size * (iter+1))/len(train_data)
        
        torch.save(model.state_dict(), f"checkpoints/checkpoint_{epoch}.pth")
        torch.save(optimizer.state_dict(), f"checkpoints/optimizer_{epoch}.pth")
        
        losses = estimate_loss()

        date_hour = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        log(f'\ncheckpoint_{date_hour} : iter {iter+1:10d} | epoch {epoch:.2f} | train loss {losses["train"]:.10f} | val loss {losses["val"]:.10f}\n')
        
        # Save the last checkpoint
        with open(f"checkpoints/checkpoint_{epoch}.info", "w") as f:
            f.write(f"iter : {iter+1}\n")
            f.write(f"epoch : {epoch}\n")
            f.write(f"losses:\n    train {losses['train']}\n    val {losses['val']}")
            f.write(f"date-hour : {date_hour}\n")
        
        # If this is the best model, save it
        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            torch.save(model.state_dict(), f"best-model.pth")
            with open(f"best-model.info", "w") as f:
                f.write(f"iter : {iter+1}\n")
                f.write(f"epoch : {epoch}\n")
                f.write(f"losses:\n    train {losses['train']}\n    val {losses['val']}")
                f.write(f"date-hour : {date_hour}\n")

if ddp:
    destroy_process_group()

if master_process:
    batch_log.close()
    log_file.close()
