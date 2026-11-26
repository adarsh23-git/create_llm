
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------- PROJECT PATHS --------

project_root = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(project_root))

from stage2.model import MiniGPT
from stage3.byte_tokenizer import ByteBPETokenizer


data_path = project_root / "data" / "tiny.txt"

tokenizer_path = (
    project_root
    / "checkpoints"
    / "byte_bpe_tokenizer.json"
)

checkpoint_path = (
    project_root
    / "checkpoints"
    / "minigpt_byte_bpe.pt"
)


# -------- DEVICE --------

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# -------- HYPERPARAMETERS --------

block_size = 64
batch_size = 16
embedding_dim = 64

max_steps = 5000
eval_interval = 100
eval_iters = 20

learning_rate = 0.001


# -------- LOAD DATASET --------

print("\nLoading Tiny Shakespeare...")

text = data_path.read_text(encoding="utf-8")

print("Dataset loaded successfully!")
print("Total characters:", len(text))


# -------- LOAD TOKENIZER --------

print("\nLoading Byte-Level BPE tokenizer...")

tokenizer = ByteBPETokenizer()

tokenizer.load(tokenizer_path)

print("Vocabulary size:", tokenizer.vocab_size)


# -------- ENCODE DATASET --------

print("\nEncoding dataset...")

encoded_data = tokenizer.encode(text)

data = torch.tensor(
    encoded_data,
    dtype=torch.long
)

print("Total tokens:", len(data))


# -------- TRAIN / VALIDATION SPLIT --------

split_index = int(0.9 * len(data))

train_data = data[:split_index]
val_data = data[split_index:]

print("\nTraining tokens:", len(train_data))
print("Validation tokens:", len(val_data))


# -------- RANDOM BATCH FUNCTION --------

def get_batch(split):

    source = train_data if split == "train" else val_data

    # Random starting positions
    ix = torch.randint(
        len(source) - block_size - 1,
        (batch_size,)
    )

    # Input sequences
    x = torch.stack([
        source[i:i + block_size]
        for i in ix
    ])

    # Next-token targets
    y = torch.stack([
        source[i + 1:i + block_size + 1]
        for i in ix
    ])

    x = x.to(device)
    y = y.to(device)

    return x, y


# -------- CREATE MODEL --------

print("\nCreating Byte-Level BPE MiniGPT...")

model = MiniGPT(
    vocab_size=tokenizer.vocab_size,
    block_size=block_size,
    embedding_dim=embedding_dim
)

model = model.to(device)

print("Model created successfully!")


# -------- LOSS FUNCTION --------

loss_function = nn.CrossEntropyLoss()


# -------- OPTIMIZER --------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


# -------- LEARNING RATE SCHEDULER --------

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=max_steps,
    eta_min=0.0001
)


# -------- EVALUATION FUNCTION --------

@torch.no_grad()
def estimate_loss():

    model.eval()

    results = {}

    for split in ["train", "val"]:

        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):

            x, y = get_batch(split)

            logits = model(x)

            loss = loss_function(
                logits.view(-1, tokenizer.vocab_size),
                y.view(-1)
            )

            losses[k] = loss.item()

        results[split] = losses.mean().item()

    model.train()

    return results


# -------- TRAINING LOOP --------

print("\nStarting training...\n")

model.train()

for step in range(max_steps):

    # Evaluate model periodically
    if step % eval_interval == 0:

        losses = estimate_loss()

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Step [{step}/{max_steps}] | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {losses['train']:.4f} | "
            f"Validation Loss: {losses['val']:.4f}"
        )

    # Get random training batch
    x, y = get_batch("train")

    # Forward pass
    logits = model(x)

    # Calculate loss
    loss = loss_function(
        logits.view(-1, tokenizer.vocab_size),
        y.view(-1)
    )

    # Clear previous gradients
    optimizer.zero_grad(set_to_none=True)

    # Backpropagation
    loss.backward()

    # Update model parameters
    optimizer.step()

    # Update learning rate
    scheduler.step()


# -------- FINAL EVALUATION --------

print("\nFinal evaluation...")

final_losses = estimate_loss()

print("Final Train Loss:", round(final_losses["train"], 4))
print("Final Validation Loss:", round(final_losses["val"], 4))


# -------- SAVE CHECKPOINT --------

torch.save(
    {
        "model_state_dict": model.state_dict(),

        "vocab_size": tokenizer.vocab_size,

        "block_size": block_size,

        "embedding_dim": embedding_dim,

        "tokenizer_type": "Byte-Level BPE",

        "tokenizer_path": str(tokenizer_path)
    },
    checkpoint_path
)

print("\nModel checkpoint saved successfully!")

print("Checkpoint location:")
print(checkpoint_path)

print("\nByte-Level BPE MiniGPT training completed!")
