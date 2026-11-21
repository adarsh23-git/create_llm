from pathlib import Path
from bpe_tokenizer import BPETokenizer


# 1. Find project root
project_root = Path(__file__).resolve().parents[1]

# 2. Load saved tokenizer
tokenizer_path = project_root / "checkpoints" / "bpe_tokenizer.json"

tokenizer = BPETokenizer()
tokenizer.load(tokenizer_path)

# 3. Read original text
data_path = project_root / "data" / "tiny.txt"

with open(data_path, "r", encoding="utf-8") as file:
    text = file.read()

# 4. Select a small sample
sample = text[:200]

# 5. Encode and decode
encoded = tokenizer.encode(sample)
decoded = tokenizer.decode(encoded)

# 6. Display results
print("\nOriginal text:")
print(sample)
                          
print("\nEncoded token IDs:")
print(encoded)

print("\nOriginal character count:", len(sample))
print("Encoded token count:", len(encoded))

print("\nDecoded text:")
print(decoded)

print("\nRound trip successful:", sample == decoded)