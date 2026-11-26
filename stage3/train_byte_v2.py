
from pathlib import Path

from byte_tokenizer_v2 import ByteBPETokenizerV2


# Project paths
project_root = Path(__file__).resolve().parents[1]

data_path = project_root / "data" / "tiny.txt"

checkpoint_dir = project_root / "checkpoints"

tokenizer_path = (
    checkpoint_dir / "byte_bpe_tokenizer_v2.json"
)

checkpoint_dir.mkdir(exist_ok=True)


# Load training text
text = data_path.read_text(encoding="utf-8")

print("Dataset loaded successfully!")
print("Original characters:", len(text))


# Create tokenizer
tokenizer = ByteBPETokenizerV2()

num_merges = 500


# Train tokenizer
final_tokens = tokenizer.train(
    text,
    num_merges=num_merges
)


# Calculate compression
original_bytes = len(text.encode("utf-8"))

final_token_count = len(final_tokens)

compression_ratio = (
    original_bytes / final_token_count
)


# Save tokenizer
tokenizer.save(tokenizer_path)


# Display results
print("\n---------- RESULTS ----------")

print("Original UTF-8 bytes:", original_bytes)

print("Final token count:", final_token_count)

print("Vocabulary size:", tokenizer.vocab_size)

print("Merge rules learned:", len(tokenizer.merges))

print("Compression ratio:", round(compression_ratio, 2))

print("\nTokenizer saved at:")
print(tokenizer_path)

print("\nByte-Level BPE v2 training completed!")
