
from pathlib import Path

from byte_tokenizer_v2 import ByteBPETokenizerV2


# Project paths
project_root = Path(__file__).resolve().parents[1]

data_path = project_root / "data" / "tiny.txt"

tokenizer_path = (
    project_root / "checkpoints" / "byte_bpe_tokenizer_v2.json"
)


# Load original dataset
text = data_path.read_text(encoding="utf-8")


# Load trained tokenizer
tokenizer = ByteBPETokenizerV2()

tokenizer.load(tokenizer_path)


# Encode complete dataset
encoded_tokens = tokenizer.encode(text)


# Decode tokens
decoded_text = tokenizer.decode(encoded_tokens)


# Display results
print("\n---------- VERIFICATION ----------")

print("Original characters:", len(text))

print("Encoded tokens:", len(encoded_tokens))

print("Vocabulary size:", tokenizer.vocab_size)

print("Decoded characters:", len(decoded_text))

print("Round trip:", text == decoded_text)


if text == decoded_text:

    print("\nByte-Level BPE v2 verification successful!")

else:

    print("\nERROR: Original and decoded text do not match!")
