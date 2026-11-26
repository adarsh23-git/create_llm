
from pathlib import Path
import sys

import torch
import torch.nn as nn

# -------- PROJECT PATHS --------

project_root = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(project_root))

from stage2.model import MiniGPT
from stage3.byte_tokenizer import ByteBPETokenizer


checkpoint_path = (
    project_root
    / "checkpoints"
    / "minigpt_byte_bpe_v2.pt"
)

tokenizer_path = (
    project_root
    / "checkpoints"
    / "byte_bpe_tokenizer.json"
)

data_path = project_root / "data" / "tiny.txt"


# -------- DEVICE --------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)


# -------- LOAD TOKENIZER --------

tokenizer = ByteBPETokenizer()
tokenizer.load(tokenizer_path)


# -------- LOAD DATA --------

text = data_path.read_text(encoding="utf-8")

encoded = tokenizer.encode(text)

data = torch.tensor(
    encoded,
    dtype=torch.long
)


# -------- LOAD CHECKPOINT --------

checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)

vocab_size = checkpoint["vocab_size"]
block_size = checkpoint["block_size"]
embedding_dim = checkpoint["embedding_dim"]


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

print("\nModel loaded successfully!")


# -------- CREATE FIXED BATCH --------

batch_size = 16

torch.manual_seed(42)

ix = torch.randint(
    0,
    10000 - block_size - 1,
    (batch_size,)
)

x = torch.stack([
    data[i:i + block_size]
    for i in ix
]).to(device)

y = torch.stack([
    data[i + 1:i + block_size + 1]
    for i in ix
]).to(device)


# -------- LOSS FUNCTION --------

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0003
)


# -------- EVALUATE FIXED BATCH --------

@torch.no_grad()
def evaluate():

    model.eval()

    logits = model(x)

    loss = loss_function(
        logits.reshape(-1, vocab_size),
        y.reshape(-1)
    )

    return loss.item()


# -------- INITIAL LOSS --------

initial_loss = evaluate()

print("\nInitial loss:", round(initial_loss, 4))


# -------- OVERFITTING TEST --------

print("\nStarting overfitting test...\n")

for step in range(501):

    if step % 100 == 0:

        current_loss = evaluate()

        print(
            f"Step [{step}/500] | "
            f"Loss: {current_loss:.4f}"
        )

    if step == 500:
        break

    model.train()

    logits = model(x)

    loss = loss_function(
        logits.reshape(-1, vocab_size),
        y.reshape(-1)
    )

    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    optimizer.step()


# -------- FINAL RESULT --------

final_loss = evaluate()

print("\nInitial loss:", round(initial_loss, 4))
print("Final loss:", round(final_loss, 4))

print("\nOverfitting diagnostic completed!")
