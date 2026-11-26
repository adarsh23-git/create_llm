
from pathlib import Path
import sys
import json

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

BEST_MODEL_PATH = CHECKPOINT_DIR / "minigpt_byte_bpe_best.pt"

HISTORY_PATH = Path(__file__).resolve().parent / "loss_history.json"

CHECKPOINT_DIR.mkdir(exist_ok=True)


# ---------------- SETTINGS ----------------

BATCH_SIZE = 16

BLOCK_SIZE = 64

EMBEDDING_DIM = 64

MAX_STEPS = 5000

EVAL_INTERVAL = 100

EVAL_ITERS = 20

LEARNING_RATE = 0.001

PATIENCE = 10

MIN_DELTA = 0.001

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

def get_batch(source):

    indices = torch.randint(
        0,
        len(source) - BLOCK_SIZE,
        (BATCH_SIZE,)
    )

    x = torch.stack([
        source[i:i + BLOCK_SIZE]
        for i in indices
    ])

    y = torch.stack([
        source[i + 1:i + BLOCK_SIZE + 1]
        for i in indices
    ])

    return x.to(DEVICE), y.to(DEVICE)


# ==================================================
#                 ESTIMATE LOSS
# ==================================================

@torch.no_grad()
def estimate_loss(model, train_data, val_data, vocab_size):

    was_training = model.training

    model.eval()

    results = {}

    for name, source in [
        ("train", train_data),
        ("val", val_data)
    ]:

        losses = torch.zeros(EVAL_ITERS)

        for k in range(EVAL_ITERS):

            x, y = get_batch(source)

            logits = model(x)

            loss = nn.functional.cross_entropy(
                logits.reshape(-1, vocab_size),
                y.reshape(-1)
            )

            losses[k] = loss.item()

        results[name] = losses.mean().item()

    model.train(was_training)

    return results


# ==================================================
#                 SAVE BEST CHECKPOINT
# ==================================================

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    vocab_size,
    train_loss,
    val_loss,
    step
):

    checkpoint = {

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "scheduler_state_dict": scheduler.state_dict(),

        "vocab_size": vocab_size,

        "block_size": BLOCK_SIZE,

        "embedding_dim": EMBEDDING_DIM,

        "tokenizer_type": "byte_bpe_final",

        "train_loss": train_loss,

        "val_loss": val_loss,

        "training_steps": step,

        "best_step": step

    }

    torch.save(
        checkpoint,
        BEST_MODEL_PATH
    )


# ==================================================
#                 SAVE LOSS HISTORY
# ==================================================

def save_history(history):

    with open(HISTORY_PATH, "w", encoding="utf-8") as file:

        json.dump(
            history,
            file,
            indent=4
        )


# ==================================================
#                 TRAIN MINIGPT
# ==================================================

def train_model():

    text = load_data()

    tokenizer = load_tokenizer()

    encoded = tokenizer.encode(text)

    data = torch.tensor(
        encoded,
        dtype=torch.long
    )

    # 90% training and 10% validation

    split_index = int(0.9 * len(data))

    train_data = data[:split_index]

    val_data = data[split_index:]

    if len(train_data) <= BLOCK_SIZE or len(val_data) <= BLOCK_SIZE:

        raise ValueError(
            "Training or validation dataset is too small."
        )

    vocab_size = tokenizer.vocab_size

    print("\nTraining tokens:", len(train_data))

    print("Validation tokens:", len(val_data))

    # ---------------- CREATE MODEL ----------------

    model = MiniGPT(
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE,
        embedding_dim=EMBEDDING_DIM
    ).to(DEVICE)

    print("\nMiniGPT created successfully!")

    # ---------------- OPTIMIZER ----------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # ---------------- SCHEDULER ----------------

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=MAX_STEPS,
        eta_min=0.0001
    )

    # ---------------- EARLY STOPPING VARIABLES ----------------

    best_val_loss = float("inf")

    best_step = 0

    patience_counter = 0

    history = []

    print("\nStarting training...\n")

    # ==================================================
    #                 TRAINING LOOP
    # ==================================================

    for step in range(MAX_STEPS):

        model.train()

        x, y = get_batch(train_data)

        logits = model(x)

        loss = nn.functional.cross_entropy(
            logits.reshape(-1, vocab_size),
            y.reshape(-1)
        )

        optimizer.zero_grad(set_to_none=True)

        loss.backward()

        optimizer.step()

        scheduler.step()

        # ---------------- EVALUATION ----------------

        if (step + 1) % EVAL_INTERVAL == 0:

            losses = estimate_loss(
                model,
                train_data,
                val_data,
                vocab_size
            )

            train_loss = losses["train"]

            val_loss = losses["val"]

            print(
                f"Step [{step + 1}/{MAX_STEPS}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Validation Loss: {val_loss:.4f}"
            )

            # ---------------- SAVE HISTORY ----------------

            history.append({

                "step": step + 1,

                "train_loss": train_loss,

                "val_loss": val_loss

            })

            # ---------------- CHECK IMPROVEMENT ----------------

            if val_loss < best_val_loss - MIN_DELTA:

                best_val_loss = val_loss

                best_step = step + 1

                patience_counter = 0

                # Save the best model

                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    vocab_size,
                    train_loss,
                    val_loss,
                    best_step
                )

                print(
                    "Best checkpoint saved! "
                    f"Validation Loss: {best_val_loss:.4f}"
                )

            else:

                patience_counter += 1

                print(
                    f"No sufficient improvement. "
                    f"Patience: {patience_counter}/{PATIENCE}"
                )

            # ---------------- EARLY STOPPING ----------------

            if patience_counter >= PATIENCE:

                print("\nEarly stopping triggered!")

                print(
                    "Training stopped because validation "
                    "loss did not improve sufficiently."
                )

                break

    # ==================================================
    #                 FINAL RESULTS
    # ==================================================

    save_history(history)

    print("\n========== FINAL RESULTS ==========")

    print("Best validation loss:", round(best_val_loss, 4))

    print("Best training step:", best_step)

    print("Total evaluations:", len(history))

    print("\nBest checkpoint saved at:")

    print(BEST_MODEL_PATH)

    print("\nLoss history saved at:")

    print(HISTORY_PATH)

    print("\nTraining completed!")


# ==================================================
#                 MAIN
# ==================================================

if __name__ == "__main__":

    print("Device:", DEVICE)

    print("\nMiniGPT - Best Checkpoint and Early Stopping")

    train_model()
