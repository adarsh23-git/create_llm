from pathlib import Path
import sys

# Project root
project_root = Path(__file__).resolve().parents[1]

# Add project root to Python module search path
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn

from stage2.model import MiniGPT
from byte_tokenizer_v2 import ByteBPETokenizerV2


# ---------------- PROJECT PATHS ----------------

project_root = Path(__file__).resolve().parents[1]

data_path = project_root / "data" / "tiny.txt"

tokenizer_path = (
    project_root / "checkpoints" / "byte_bpe_tokenizer_v2.json"
)

checkpoint_dir = project_root / "checkpoints"

model_path = (
    checkpoint_dir / "minigpt_byte_bpe_v3.pt"
)

checkpoint_dir.mkdir(exist_ok=True)


# ---------------- DEVICE ----------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)


# ---------------- LOAD DATASET ----------------

text = data_path.read_text(encoding="utf-8")

print("Dataset loaded successfully!")


# ---------------- LOAD TOKENIZER ----------------

tokenizer = ByteBPETokenizerV2()

tokenizer.load(tokenizer_path)

print("Tokenizer loaded successfully!")


# Convert text into token IDs
encoded_data = tokenizer.encode(text)

data = torch.tensor(
    encoded_data,
    dtype=torch.long
)

print("Total tokens:", len(data))
print("Vocabulary size:", tokenizer.vocab_size)


# ---------------- TRAIN / VALIDATION SPLIT ----------------

split_index = int(0.9 * len(data))

train_data = data[:split_index]

val_data = data[split_index:]

print("Training tokens:", len(train_data))
print("Validation tokens:", len(val_data))


# ---------------- MODEL CONFIGURATION ----------------

batch_size = 16

block_size = 64

embedding_dim = 64

vocab_size = tokenizer.vocab_size

max_steps = 5000

eval_interval = 100

eval_iters = 20


# ---------------- CREATE MODEL ----------------

model = MiniGPT(
    vocab_size=vocab_size,
    block_size=block_size,
    embedding_dim=embedding_dim
).to(device)

print("\nModel created successfully!")


# ---------------- BATCH GENERATION ----------------

def get_batch(split):

    source = train_data if split == "train" else val_data

    # Random starting positions
    indices = torch.randint(
        0,
        len(source) - block_size - 1,
        (batch_size,)
    )

    # Input sequences
    x = torch.stack([
        source[i:i + block_size]
        for i in indices
    ])

    # Next-token targets
    y = torch.stack([
        source[i + 1:i + block_size + 1]
        for i in indices
    ])

    x = x.to(device)
    y = y.to(device)

    return x, y


# ---------------- LOSS EVALUATION ----------------

@torch.no_grad()
def estimate_loss():

    model.eval()

    results = {}

    for split in ["train", "val"]:

        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):

            x, y = get_batch(split)

            logits = model(x)

            loss = nn.functional.cross_entropy(
                logits.view(-1, vocab_size),
                y.view(-1)
            )

            losses[k] = loss.item()

        results[split] = losses.mean().item()

    model.train()

    return results


# ---------------- OPTIMIZER ----------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=max_steps,
    eta_min=0.0001
)


# ---------------- INITIAL LOSS ----------------

initial_losses = estimate_loss()

print("\nInitial Train Loss:", round(initial_losses["train"], 4))

print("Initial Validation Loss:", round(initial_losses["val"], 4))


# ---------------- TRAINING LOOP ----------------

print("\nStarting training...\n")

for step in range(max_steps):

    model.train()

    # Get training batch
    x, y = get_batch("train")

    # Forward pass
    logits = model(x)

    # Calculate loss
    loss = nn.functional.cross_entropy(
        logits.view(-1, vocab_size),
        y.view(-1)
    )

    # Clear old gradients
    optimizer.zero_grad(set_to_none=True)

    # Backpropagation
    loss.backward()

    # Update model parameters
    optimizer.step()

    # Update learning rate
    scheduler.step()

    # Evaluate periodically
    if (step + 1) % eval_interval == 0:

        losses = estimate_loss()

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Step [{step+1}/{max_steps}] | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {losses['train']:.4f} | "
            f"Validation Loss: {losses['val']:.4f}"
        )


# ---------------- FINAL EVALUATION ----------------

final_losses = estimate_loss()

print("\n---------- FINAL RESULTS ----------")

print(
    "Final Train Loss:",
    round(final_losses["train"], 4)
)

print(
    "Final Validation Loss:",
    round(final_losses["val"], 4)
)


# ---------------- SAVE MODEL CHECKPOINT ----------------

checkpoint = {

    "model_state_dict": model.state_dict(),

    "optimizer_state_dict": optimizer.state_dict(),

    "vocab_size": vocab_size,

    "block_size": block_size,

    "embedding_dim": embedding_dim,

    "tokenizer_type": "byte_bpe_v2",

    "tokenizer_path": str(tokenizer_path),

    "training_steps": max_steps,

    "train_loss": final_losses["train"],

    "val_loss": final_losses["val"]

}

torch.save(
    checkpoint,
    model_path
)

print("\nModel checkpoint saved successfully!")

print("Checkpoint location:")
print(model_path)

print("\nByte-Level MiniGPT v3 training completed!")

