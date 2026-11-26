
from pathlib import Path

from byte_tokenizer import ByteBPETokenizer


# -------- PROJECT PATHS --------

project_root = Path(__file__).resolve().parents[1]

data_path = project_root / "data" / "tiny.txt"

tokenizer_path = (
    project_root
    / "checkpoints"
    / "byte_bpe_tokenizer.json"
)


# -------- LOAD DATASET --------

text = data_path.read_text(encoding="utf-8")


# -------- LOAD TOKENIZER --------

tokenizer = ByteBPETokenizer()

tokenizer.load(tokenizer_path)


# -------- TEST ENCODING --------

encoded_data = tokenizer.encode(text)

print("Original characters:", len(text))
print("Encoded tokens:", len(encoded_data))
print("Vocabulary size:", tokenizer.vocab_size)


# -------- TEST DECODING --------

decoded_text = tokenizer.decode(encoded_data)

print("\nDecoded characters:", len(decoded_text))

print("Round trip:", decoded_text == text)


# -------- FINAL CHECK --------

if decoded_text == text:
    print("\nByte-Level BPE tokenizer verification successful!")

else:
    print("\nTokenizer verification failed!")
