
from pathlib import Path

from byte_tokenizer import ByteBPETokenizer


# -------- PROJECT PATHS --------

project_root = Path(__file__).resolve().parents[1]

data_path = project_root / "data" / "tiny.txt"

checkpoint_dir = project_root / "checkpoints"

tokenizer_path = checkpoint_dir / "byte_bpe_tokenizer.json"


# Create checkpoint directory if it does not exist
checkpoint_dir.mkdir(exist_ok=True)


# -------- LOAD DATASET --------

print("Loading Tiny Shakespeare dataset...")

text = data_path.read_text(encoding="utf-8")

print("Dataset loaded successfully!")

print("Total characters:", len(text))


# -------- TRAIN BYTE BPE TOKENIZER --------

tokenizer = ByteBPETokenizer()

num_merges = 500

print("\nStarting Byte-Level BPE training...")

final_tokens = tokenizer.train(
    text,
    num_merges=num_merges
)


# -------- TOKEN STATISTICS --------

original_bytes = len(text.encode("utf-8"))

final_token_count = len(final_tokens)

compression_ratio = original_bytes / final_token_count

print("\n-------- TOKENIZER STATISTICS --------")

print("Original UTF-8 bytes:", original_bytes)

print("Final token count:", final_token_count)

print("Vocabulary size:", tokenizer.vocab_size)

print("Merge rules learned:", len(tokenizer.merges))

print("Compression ratio:", round(compression_ratio, 2))


# -------- SAVE TOKENIZER --------

tokenizer.save(tokenizer_path)

print("\nTokenizer saved at:")

print(tokenizer_path)

print("\nByte-Level BPE training completed!")
