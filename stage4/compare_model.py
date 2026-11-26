from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from pathlib import Path

import torch

from stage2.model import MiniGPT
from stage3.byte_bpe_complete import ByteBPETokenizer


# -----------------------------------------
# 1. PROJECT CONFIGURATION
# -----------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TOKENIZER_PATH = PROJECT_ROOT / "checkpoints" / "byte_bpe_tokenizer_final.json"

ORIGINAL_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "minigpt_byte_bpe_final.pt"

BEST_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "minigpt_byte_bpe_best.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)


# -----------------------------------------
# 2. LOAD TOKENIZER
# -----------------------------------------

tokenizer = ByteBPETokenizer()

tokenizer.load(str(TOKENIZER_PATH))

print("Tokenizer loaded successfully!")


# -----------------------------------------
# 3. LOAD MODEL FROM CHECKPOINT
# -----------------------------------------

def load_model(checkpoint_path):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True
    )

    model = MiniGPT(
        vocab_size=checkpoint["vocab_size"],
        block_size=checkpoint["block_size"],
        embedding_dim=checkpoint["embedding_dim"]
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    return model, checkpoint


# -----------------------------------------
# 4. LOAD BOTH MODELS
# -----------------------------------------

original_model, original_checkpoint = load_model(
    ORIGINAL_CHECKPOINT
)

best_model, best_checkpoint = load_model(
    BEST_CHECKPOINT
)

print("Both models loaded successfully!")


# -----------------------------------------
# 5. GENERATE TEXT
# -----------------------------------------


def generate_text(model, prompt, block_size, max_new_tokens=300):

    # Convert prompt into token IDs
    encoded = tokenizer.encode(prompt)

    input_ids = torch.tensor(
        [encoded],
        dtype=torch.long,
        device=DEVICE
    )

    model.eval()

    with torch.no_grad():

        for _ in range(max_new_tokens):

            # Keep only the most recent context tokens
            input_context = input_ids[:, -block_size:]

            # Get predictions from the model
            logits = model(input_context)

            # Select predictions for the final position
            logits = logits[:, -1, :]

            # Apply temperature
            logits = logits / 0.8

            # Apply Top-K sampling
            top_k = 50

            values, _ = torch.topk(
                logits,
                min(top_k, logits.size(-1))
            )

            logits[logits < values[:, [-1]]] = float("-inf")

            # Apply Top-P sampling
            top_p = 0.90

            sorted_logits, sorted_indices = torch.sort(
                logits,
                descending=True
            )

            sorted_probs = torch.softmax(
                sorted_logits,
                dim=-1
            )

            cumulative_probs = torch.cumsum(
                sorted_probs,
                dim=-1
            )

            sorted_indices_to_remove = cumulative_probs > top_p

            # Keep at least the highest-probability token
            sorted_indices_to_remove[..., 1:] = (
                sorted_indices_to_remove[..., :-1].clone()
            )

            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = torch.zeros_like(
                logits,
                dtype=torch.bool
            ).scatter(
                1,
                sorted_indices,
                sorted_indices_to_remove
            )

            logits[indices_to_remove] = float("-inf")

            # Convert logits into probabilities
            probabilities = torch.softmax(
                logits,
                dim=-1
            )

            # Sample the next token
            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            # Append the new token to the sequence
            input_ids = torch.cat(
                (input_ids, next_token),
                dim=1
            )

    # Convert generated token IDs back into text
    decoded_text = tokenizer.decode(input_ids[0].tolist())

    print("\nDEBUG INFORMATION")
    print("Original prompt:", repr(prompt))
    print("Encoded prompt:", encoded)
    print("Prompt token count:", len(encoded))
    print("Total generated token count:", input_ids.shape[1])
    print("Decoded text:", repr(decoded_text))

    return decoded_text


# -----------------------------------------
# 6. SAME PROMPT FOR BOTH MODELS
# -----------------------------------------

prompt = "First Citizen:"


print("\n========== ORIGINAL MODEL ==========")

generate_text(
    original_model,
    prompt,
    original_checkpoint["block_size"]
)


print("\n========== BEST MODEL ==========")

generate_text(
    original_model,
    prompt,
    original_checkpoint["block_size"]
)


# -----------------------------------------
# 7. CHECKPOINT INFORMATION
# -----------------------------------------

print("\n========== CHECKPOINT COMPARISON ==========")

print(
    "Original training steps:",
    original_checkpoint["training_steps"]
)

print(
    "Best model training steps:",
    best_checkpoint["training_steps"]
)

print(
    "Original validation loss:",
    original_checkpoint["val_loss"]
)

print(
    "Best validation loss:",
    best_checkpoint["val_loss"]
)

print("\nComparison completed!")
