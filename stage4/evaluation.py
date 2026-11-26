
from pathlib import Path
import sys

import torch
import torch.nn as nn

# ---------------- PROJECT PATHS ----------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from stage2.model import MiniGPT
from stage3.byte_bpe_complete import ByteBPETokenizer

DATA_PATH = PROJECT_ROOT / "data" / "tiny.txt"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

TOKENIZER_PATH = CHECKPOINT_DIR / "byte_bpe_tokenizer_final.json"

MODEL_PATH = CHECKPOINT_DIR / "minigpt_byte_bpe_final.pt"


# ---------------- SETTINGS ----------------

BATCH_SIZE = 16

EVAL_ITERS = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==================================================
#                 LOAD DATA
# ==================================================

def load_data():

    text = DATA_PATH.read_text(encoding="utf-8")

    print("Dataset loaded successfully!")

    return text


# ==================================================
#                 LOAD TOKENIZER
# ==================================================

def load_tokenizer():

    tokenizer = ByteBPETokenizer()

    tokenizer.load(TOKENIZER_PATH)

    return tokenizer


# ==================================================
#                 GET BATCH
# ==================================================

def get_batch(source, block_size):

    if len(source) <= block_size:

        raise ValueError(
            "Dataset is too small for the selected block size."
        )

    indices = torch.randint(
        0,
        len(source) - block_size,
        (BATCH_SIZE,)
    )

    x = torch.stack([
        source[i:i + block_size]
        for i in indices
    ])

    y = torch.stack([
        source[i + 1:i + block_size + 1]
        for i in indices
    ])

    return x.to(DEVICE), y.to(DEVICE)


# ==================================================
#                 EVALUATE LOSS
# ==================================================

@torch.no_grad()
def estimate_loss(model, source, block_size, vocab_size):

    was_training = model.training

    model.eval()

    losses = torch.zeros(EVAL_ITERS)

    for k in range(EVAL_ITERS):

        x, y = get_batch(source, block_size)

        logits = model(x)

        loss = nn.functional.cross_entropy(
            logits.reshape(-1, vocab_size),
            y.reshape(-1)
        )

        losses[k] = loss.item()

    model.train(was_training)

    return losses.mean().item()


# ==================================================
#                 MAIN EVALUATION
# ==================================================

def evaluate():

    print("Device:", DEVICE)

    print("\nLoading dataset...")

    text = load_data()

    print("\nLoading tokenizer...")

    tokenizer = load_tokenizer()

    encoded = tokenizer.encode(text)

    data = torch.tensor(
        encoded,
        dtype=torch.long
    )

    # Use the same 90/10 split as Stage 3

    split_index = int(0.9 * len(data))

    train_data = data[:split_index]

    val_data = data[split_index:]

    print("\nLoading checkpoint...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    vocab_size = checkpoint["vocab_size"]

    block_size = checkpoint["block_size"]

    embedding_dim = checkpoint["embedding_dim"]

    model = MiniGPT(
        vocab_size=vocab_size,
        block_size=block_size,
        embedding_dim=embedding_dim
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print("Model loaded successfully!")

    print("\n---------- MODEL INFORMATION ----------")

    print("Vocabulary size:", vocab_size)

    print("Block size:", block_size)

    print("Embedding dimension:", embedding_dim)

    print("Training steps:", checkpoint["training_steps"])

    print("\n---------- STORED LOSSES ----------")

    print(
        "Stored train loss:",
        round(checkpoint["train_loss"], 4)
    )

    print(
        "Stored validation loss:",
        round(checkpoint["val_loss"], 4)
    )

    print("\nStarting fresh evaluation...\n")

    train_loss = estimate_loss(
        model,
        train_data,
        block_size,
        vocab_size
    )

    val_loss = estimate_loss(
        model,
        val_data,
        block_size,
        vocab_size
    )

    # Calculate generalization gap

    gap = val_loss - train_loss

    print("\n========== EVALUATION RESULTS ==========")

    print("Training loss:", round(train_loss, 4))

    print("Validation loss:", round(val_loss, 4))

    print("Generalization gap:", round(gap, 4))

    print("\n---------- DIAGNOSIS ----------")

    if gap > 0.5:

        print("Possible overfitting detected.")

        print("Validation loss is noticeably higher.")

    elif gap > 0:

        print("Small generalization gap detected.")

        print("Monitor validation loss during training.")

    else:

        print("No positive generalization gap detected.")

        print("Validation loss is not higher than training loss.")

    print("\nEvaluation completed!")


# ==================================================
#                 RUN
# ==================================================

if __name__ == "__main__":

    evaluate()
