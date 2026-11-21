from pathlib import Path
from bpe_tokenizer import BPETokenizer


# 1. Find project root
project_root = Path(__file__).resolve().parents[1]

# 2. Load training text
data_path = project_root / "data" / "tiny.txt"

with open(data_path, "r", encoding="utf-8") as file:
    text = file.read()

print("Total characters:", len(text))

# 3. Create tokenizer
tokenizer = BPETokenizer()

# 4. Train BPE
final_tokens = tokenizer.train(
    text,
    num_merges=500
)

# 5. Build vocabulary
vocab = tokenizer.build_vocab()

print("Vocabulary size:", len(vocab))
print("Total merge rules:", len(tokenizer.merges))

# 6. Save tokenizer
save_path = project_root / "checkpoints" / "bpe_tokenizer.json"

save_path.parent.mkdir(parents=True, exist_ok=True)

tokenizer.save(save_path)