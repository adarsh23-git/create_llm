
from pathlib import Path
import sys
import json
import re

import torch
import torch.nn as nn

# ---------------- PROJECT PATHS ----------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from stage2.model import MiniGPT

DATA_PATH = PROJECT_ROOT / "data" / "tiny.txt"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

TOKENIZER_PATH = CHECKPOINT_DIR / "byte_bpe_tokenizer_final.json"

MODEL_PATH = CHECKPOINT_DIR / "minigpt_byte_bpe_final.pt"

CHECKPOINT_DIR.mkdir(exist_ok=True)

# ---------------- SETTINGS ----------------
MODE = "generate"

# Available modes:
# train_tokenizer
# verify
# train_model
# generate

NUM_MERGES = 500

BATCH_SIZE = 16

BLOCK_SIZE = 64

EMBEDDING_DIM = 64

MAX_STEPS = 5000

EVAL_INTERVAL = 100

EVAL_ITERS = 20

LEARNING_RATE = 0.001

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==================================================
#                 BYTE BPE TOKENIZER
# ==================================================

class ByteBPETokenizer:

    def __init__(self):

        self.vocab = {}

        self.itos = {}

        self.merges = []

        self.vocab_size = 0

        self.reset_vocab()

    # ---------------- INITIAL VOCABULARY ----------------

    def reset_vocab(self):

        self.vocab = {
            bytes([i]): i
            for i in range(256)
        }

        self.itos = {
            index: token
            for token, index in self.vocab.items()
        }

        self.merges = []

        self.vocab_size = 256

    # ---------------- PRE TOKENIZATION ----------------

    def pre_tokenize(self, text):

        # Split text into words, punctuation and whitespace
        pieces = re.findall(
            r"\w+|[^\w\s]|\s+",
            text
        )

        return pieces

    # ---------------- BYTE ENCODING ----------------

    def encode_bytes(self, text):

        return [
            bytes([byte])
            for byte in text.encode("utf-8")
        ]

    # ---------------- PAIR OPERATIONS ----------------

    def get_pair_counts(self, sequences):

        pair_counts = {}

        # Count pairs independently inside each piece
        for tokens in sequences:

            for i in range(len(tokens) - 1):

                pair = (
                    tokens[i],
                    tokens[i + 1]
                )

                pair_counts[pair] = (
                    pair_counts.get(pair, 0) + 1
                )

        return pair_counts

    def merge_pair(self, tokens, pair):

        merged_tokens = []

        i = 0

        while i < len(tokens):

            if (
                i < len(tokens) - 1
                and tokens[i] == pair[0]
                and tokens[i + 1] == pair[1]
            ):

                merged_tokens.append(
                    tokens[i] + tokens[i + 1]
                )

                i += 2

            else:

                merged_tokens.append(tokens[i])

                i += 1

        return merged_tokens

    # ---------------- TRAIN TOKENIZER ----------------

    def train(self, text, num_merges):

        self.reset_vocab()

        pieces = self.pre_tokenize(text)

        # Every piece has its own byte-token sequence
        sequences = [
            self.encode_bytes(piece)
            for piece in pieces
        ]

        for step in range(num_merges):

            pair_counts = self.get_pair_counts(sequences)

            if not pair_counts:
                break

            most_frequent_pair = max(
                pair_counts,
                key=pair_counts.get
            )

            # Merge independently inside every piece
            sequences = [
                self.merge_pair(sequence, most_frequent_pair)
                for sequence in sequences
            ]

            new_token = (
                most_frequent_pair[0]
                + most_frequent_pair[1]
            )

            new_id = len(self.vocab)

            self.vocab[new_token] = new_id

            self.itos[new_id] = new_token

            self.merges.append(most_frequent_pair)

            print(
                f"Merge {step + 1}: "
                f"{most_frequent_pair} -> {new_token}"
            )

        self.vocab_size = len(self.vocab)

        # Flatten final sequences for compression statistics
        final_tokens = [
            token
            for sequence in sequences
            for token in sequence
        ]

        return final_tokens

    # ---------------- TEXT ENCODING ----------------

    def encode(self, text):

        pieces = self.pre_tokenize(text)

        token_ids = []

        for piece in pieces:

            tokens = self.encode_bytes(piece)

            # Apply learned merges in their original order
            for pair in self.merges:

                tokens = self.merge_pair(
                    tokens,
                    pair
                )

            for token in tokens:

                token_ids.append(
                    self.vocab[token]
                )

        return token_ids

    # ---------------- TEXT DECODING ----------------

    def decode(self, token_ids):

        byte_tokens = [
            self.itos[token_id]
            for token_id in token_ids
        ]

        byte_data = b"".join(byte_tokens)

        return byte_data.decode(
            "utf-8",
            errors="replace"
        )

    # ---------------- SAVE TOKENIZER ----------------

    def save(self, path):

        data = {
            "merges": [
                [
                    pair[0].hex(),
                    pair[1].hex()
                ]
                for pair in self.merges
            ]
        }

        with open(path, "w", encoding="utf-8") as file:

            json.dump(data, file, indent=4)

        print("Tokenizer saved successfully!")

    # ---------------- LOAD TOKENIZER ----------------

    def load(self, path):

        with open(path, "r", encoding="utf-8") as file:

            data = json.load(file)

        self.reset_vocab()

        for pair_data in data["merges"]:

            pair = (
                bytes.fromhex(pair_data[0]),
                bytes.fromhex(pair_data[1])
            )

            new_token = pair[0] + pair[1]

            new_id = len(self.vocab)

            self.vocab[new_token] = new_id

            self.itos[new_id] = new_token

            self.merges.append(pair)

        self.vocab_size = len(self.vocab)

        print("Tokenizer loaded successfully!")


# ==================================================
#                 LOAD DATA
# ==================================================

def load_text():

    text = DATA_PATH.read_text(encoding="utf-8")

    print("Dataset loaded successfully!")

    return text


def load_tokenizer():

    tokenizer = ByteBPETokenizer()

    tokenizer.load(TOKENIZER_PATH)

    return tokenizer


# ==================================================
#                 TRAIN TOKENIZER
# ==================================================

def train_tokenizer():

    text = load_text()

    tokenizer = ByteBPETokenizer()

    print("\nStarting Byte BPE training...\n")

    final_tokens = tokenizer.train(
        text,
        NUM_MERGES
    )

    original_bytes = len(text.encode("utf-8"))

    compression_ratio = (
        original_bytes / len(final_tokens)
    )

    tokenizer.save(TOKENIZER_PATH)

    print("\n---------- TOKENIZER RESULTS ----------")

    print("Original bytes:", original_bytes)

    print("Final tokens:", len(final_tokens))

    print("Vocabulary size:", tokenizer.vocab_size)

    print("Merge rules:", len(tokenizer.merges))

    print("Compression ratio:", round(compression_ratio, 2))


# ==================================================
#                 VERIFY TOKENIZER
# ==================================================

def verify_tokenizer():

    text = load_text()

    tokenizer = load_tokenizer()

    encoded = tokenizer.encode(text)

    decoded = tokenizer.decode(encoded)

    print("\n---------- VERIFICATION ----------")

    print("Original characters:", len(text))

    print("Encoded tokens:", len(encoded))

    print("Vocabulary size:", tokenizer.vocab_size)

    print("Decoded characters:", len(decoded))

    print("Round trip:", text == decoded)

    if text == decoded:

        print("\nTokenizer verification successful!")

    else:

        print("\nTokenizer verification failed!")


# ==================================================
#                 GET BATCH
# ==================================================

def get_batch(source):

    indices = torch.randint(
        0,
        len(source) - BLOCK_SIZE - 1,
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
#                 EVALUATE LOSS
# ==================================================

@torch.no_grad()
def estimate_loss(model, train_data, val_data, vocab_size):

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

    model.train()

    return results


# ==================================================
#                 TRAIN MINIGPT
# ==================================================

def train_model():

    text = load_text()

    tokenizer = load_tokenizer()

    encoded = tokenizer.encode(text)

    data = torch.tensor(
        encoded,
        dtype=torch.long
    )

    split_index = int(0.9 * len(data))

    train_data = data[:split_index]

    val_data = data[split_index:]

    vocab_size = tokenizer.vocab_size

    print("\nTraining tokens:", len(train_data))

    print("Validation tokens:", len(val_data))

    model = MiniGPT(
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE,
        embedding_dim=EMBEDDING_DIM
    ).to(DEVICE)

    print("\nMiniGPT created successfully!")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=MAX_STEPS,
        eta_min=0.0001
    )

    initial = estimate_loss(
        model,
        train_data,
        val_data,
        vocab_size
    )

    print("\nInitial train loss:", round(initial["train"], 4))

    print("Initial validation loss:", round(initial["val"], 4))

    print("\nStarting training...\n")

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

        if (step + 1) % EVAL_INTERVAL == 0:

            losses = estimate_loss(
                model,
                train_data,
                val_data,
                vocab_size
            )

            print(
                f"Step [{step + 1}/{MAX_STEPS}] | "
                f"Train Loss: {losses['train']:.4f} | "
                f"Validation Loss: {losses['val']:.4f}"
            )

    final_losses = estimate_loss(
        model,
        train_data,
        val_data,
        vocab_size
    )

    checkpoint = {

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "vocab_size": vocab_size,

        "block_size": BLOCK_SIZE,

        "embedding_dim": EMBEDDING_DIM,

        "tokenizer_type": "byte_bpe_final",

        "train_loss": final_losses["train"],

        "val_loss": final_losses["val"],

        "training_steps": MAX_STEPS

    }

    torch.save(
        checkpoint,
        MODEL_PATH
    )

    print("\n---------- FINAL RESULTS ----------")

    print("Train loss:", round(final_losses["train"], 4))

    print("Validation loss:", round(final_losses["val"], 4))

    print("\nModel checkpoint saved successfully!")

    print(MODEL_PATH)


# ==================================================
#                 GENERATE TEXT
# ==================================================

@torch.no_grad()
def generate():

    tokenizer = load_tokenizer()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model = MiniGPT(
        vocab_size=checkpoint["vocab_size"],
        block_size=checkpoint["block_size"],
        embedding_dim=checkpoint["embedding_dim"]
    ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    prompt = "First Citizen:"

    temperature = 0.8

    top_k = 50

    top_p = 0.90

    max_new_tokens = 500

    input_ids = tokenizer.encode(prompt)

    context = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=DEVICE
    )

    generated = context.clone()

    for step in range(max_new_tokens):

        current_context = generated[
            :, -checkpoint["block_size"]:
        ]

        logits = model(current_context)

        logits = logits[:, -1, :]

        logits = logits / temperature

        # Top-K filtering
        values, _ = torch.topk(
            logits,
            min(top_k, logits.size(-1))
        )

        logits[logits < values[:, [-1]]] = float("-inf")

        # Top-P filtering
        sorted_logits, sorted_indices = torch.sort(
            logits,
            descending=True
        )

        cumulative_probs = torch.cumsum(
            torch.softmax(sorted_logits, dim=-1),
            dim=-1
        )

        sorted_remove = cumulative_probs > top_p

        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()

        sorted_remove[..., 0] = False

        remove = torch.zeros_like(
            logits,
            dtype=torch.bool
        )

        remove.scatter_(
            1,
            sorted_indices,
            sorted_remove
        )

        logits[remove] = float("-inf")

        probabilities = torch.softmax(
            logits,
            dim=-1
        )

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )

        generated = torch.cat(
            (generated, next_token),
            dim=1
        )

    output = tokenizer.decode(
        generated[0].tolist()
    )

    print("\n---------- GENERATED TEXT ----------\n")

    print(output)

    print("\nGeneration completed!")


# ==================================================
#                 MAIN
# ==================================================

if __name__ == "__main__":

    print("Device:", DEVICE)

    print("\nByte-Level BPE Complete Pipeline")

    if MODE == "train_tokenizer":

        train_tokenizer()

    elif MODE == "verify":

        verify_tokenizer()

    elif MODE == "train_model":

        train_model()

    elif MODE == "generate":

        generate()

    else:

        print("Invalid MODE selected!")
