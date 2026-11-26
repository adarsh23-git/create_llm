
from pathlib import Path
import json

import matplotlib.pyplot as plt

# ---------------- PROJECT PATHS ----------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HISTORY_PATH = PROJECT_ROOT / "stage4" / "loss_history.json"

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "minigpt_byte_bpe_best.pt"


# ==================================================
#                 LOAD HISTORY
# ==================================================

with open(HISTORY_PATH, "r", encoding="utf-8") as file:

    history = json.load(file)

print("Loss history loaded successfully!")


# ==================================================
#                 EXTRACT DATA
# ==================================================

steps = [
    item["step"]
    for item in history
]

train_losses = [
    item["train_loss"]
    for item in history
]

val_losses = [
    item["val_loss"]
    for item in history
]


# ==================================================
#                 FIND BEST CHECKPOINT
# ==================================================

import torch

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True
)

best_step = checkpoint["best_step"]

best_val_loss = checkpoint["val_loss"]


# ==================================================
#                 PLOT LOSS
# ==================================================

plt.figure(figsize=(11, 6))

plt.plot(
    steps,
    train_losses,
    label="Training Loss",
    linewidth=2
)

plt.plot(
    steps,
    val_losses,
    label="Validation Loss",
    linewidth=2
)

plt.axvline(
    x=best_step,
    linestyle="--",
    label=f"Best Checkpoint (Step {best_step})"
)

plt.scatter(
    best_step,
    best_val_loss,
    label=f"Best Val Loss: {best_val_loss:.4f}"
)

plt.xlabel("Training Steps")

plt.ylabel("Loss")

plt.title("MiniGPT Training and Validation Loss")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ==================================================
#                 RESULTS
# ==================================================

print("\n---------- LOSS ANALYSIS ----------")

print("Total evaluations:", len(history))

print("Best training step:", best_step)

print("Best validation loss:", round(best_val_loss, 4))

print("\nVisualization completed!")
