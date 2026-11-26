
from pathlib import Path
import sys

import torch
import torch.nn as nn

# -------- PROJECT PATHS --------

project_root = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(project_root))

from stage2.model import MiniGPT
from stage3.byte_tokenizer import ByteBPETokenizer


data_path = project_root / "data" / "tiny.txt"

tokenizer_path = (
    project_root / "checkpoints" / "byte_bpe_tokenizer.json"
)

old_checkpoint_path = (
    project_root / "checkpoints" / "minigpt_byte_bpe.pt"
)

new_checkpoint_path = (
    project_root / "checkpoints" / "minigpt_byte_bpe_v2.pt"
)


# -------- DEVICE --------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)


# -------- HYPERPARAMETERS --------

batch_size = 16
max_steps = 5000

eval_interval = 100
eval_iters = 20

learning_rate = 0.0001


# -------- LOAD DATASET --------

print("\nLoading dataset...")

text = data_path.read_text(encoding="utf-8")

print("Dataset loaded:", len(text), "characters")


# -------- LOAD TOKENIZER --------

tokenizer = ByteBPETokenizer()
tokenizer.load(tokenizer_path)

print("Vocabulary size:", tokenizer.vocab_size)


# -------- ENCODE DATASET --------

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


# -------- LOAD OLD CHECKPOINT --------

print("\nLoading previous model...")

checkpoint = torch.load(
    old_checkpoint_path,
    map_location=device
)

block_size = checkpoint["block_size"]
embedding_dim = checkpoint["embedding_dim"]
vocab_size = checkpoint["vocab_size"]


# -------- CREATE MODEL --------

model = MiniGPT(
    vocab_size=vocab_size,
    block_size=block_size,
    embedding_dim=embedding_dim
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

print("Previous model weights loaded successfully!")


# -------- BATCH FUNCTION --------

def get_batch(split):

    source = train_data if split == "train" else val_data

    ix = torch.randint(
        len(source) - block_size - 1,
        (batch_size,)
    )

    x = torch.stack([
        source[i:i + block_size]
        for i in ix
    ])

    y = torch.stack([
        source[i + 1:i + block_size + 1]
        for i in ix
    ])

    return x.to(device), y.to(device)


# -------- LOSS FUNCTION --------

loss_function = nn.CrossEntropyLoss()


# -------- OPTIMIZER --------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


# -------- SCHEDULER --------

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=max_steps,
    eta_min=0.00001
)


# -------- EVALUATION --------

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
                logits.view(-1, vocab_size),
                y.view(-1)
            )

            losses[k] = loss.item()

        results[split] = losses.mean().item()

    model.train()

    return results


# -------- CONTINUE TRAINING --------

print("\nContinuing previous training...\n")

model.train()

for step in range(max_steps):

    if step % eval_interval == 0:

        losses = estimate_loss()

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Step [{step}/{max_steps}] | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {losses['train']:.4f} | "
            f"Validation Loss: {losses['val']:.4f}"
        )

    x, y = get_batch("train")

    logits = model(x)

    loss = loss_function(
        logits.view(-1, vocab_size),
        y.view(-1)
    )

    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    optimizer.step()

    scheduler.step()


# -------- FINAL EVALUATION --------

print("\nFinal evaluation...")

final_losses = estimate_loss()

print("Final Train Loss:", round(final_losses["train"], 4))

print(
    "Final Validation Loss:",
    round(final_losses["val"], 4)
)


# -------- SAVE NEW CHECKPOINT --------

torch.save(
    {
        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "vocab_size": vocab_size,

        "block_size": block_size,

        "embedding_dim": embedding_dim,

        "tokenizer_type": "Byte-Level BPE",

        "tokenizer_path": str(tokenizer_path),

        "training_steps": 10000
    },
    new_checkpoint_path
)

print("\nNew checkpoint saved successfully!")

print(new_checkpoint_path)

print("\nAdditional training completed!")
