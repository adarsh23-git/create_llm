import re
import json
from pathlib import Path

class BPETokenizer:
    def __init__(self):
        self.vocab = {}
        self.merges = []

    def get_pairs(self, tokens):
        pairs = []
        for i in range(len(tokens) - 1):
            pair = (
                tokens[i],
                tokens[i + 1]
            )
            pairs.append(pair)

        return pairs

    def count_pairs(self, pairs):

        pair_frequency = {}

        for pair in pairs:

            if pair in pair_frequency:
                pair_frequency[pair] += 1

            else:
                pair_frequency[pair] = 1

        return pair_frequency

    def get_most_frequent_pair(self, pair_frequency):

# max() finds the item with the highest value.
# pair_frequency contains pairs as dictionary keys.
# key=pair_frequency.get tells Python to compare their frequency values, not the pair names.

        if not pair_frequency:
            return None

        most_frequent_pair = max(
            pair_frequency,
            key=pair_frequency.get
        )

        return most_frequent_pair

    
    def merge_pair(self, tokens, pair):

        merged_tokens = []

        i = 0

        while i < len(tokens):

            if (
                i < len(tokens) - 1
                and tokens[i] == pair[0]
                and tokens[i + 1] == pair[1]
            ):

                merged_token = tokens[i] + tokens[i + 1]

                merged_tokens.append(merged_token)

                i += 2

            else:

                merged_tokens.append(tokens[i])

                i += 1

        return merged_tokens

# tokens = list(text) converts text into individual characters.
# for i in range(num_merges) repeats the merging process.
# self.get_pairs(tokens) finds the current adjacent pairs.
# self.count_pairs(pairs) recalculates their frequencies.
# self.get_most_frequent_pair(...) selects the next pair.
# self.merge_pair(...) performs the merge.
# self.merges.append(...) remembers the merge rule for future encoding.

    def train(self, text, num_merges):

        # 1. Pre-tokenize text
        pieces = self.pre_tokenize(text)

        # 2. Convert every piece into a list of characters
        tokens = [list(piece) for piece in pieces]

        # 3. Initialize vocabulary and merges
        self.vocab = set(text)
        self.merges = []

        for i in range(num_merges):

            # 4. Count pairs inside each piece
            pair_frequency = {}

            for piece_tokens in tokens:

                pairs = self.get_pairs(piece_tokens)

                piece_frequency = self.count_pairs(pairs)

                for pair, frequency in piece_frequency.items():

                    if pair in pair_frequency:
                        pair_frequency[pair] += frequency
                    else:
                        pair_frequency[pair] = frequency

            # 5. Find the most frequent pair globally
            most_frequent_pair = self.get_most_frequent_pair(pair_frequency)

            if most_frequent_pair is None:
                break

            # 6. Merge the pair separately inside every piece
            tokens = [
                self.merge_pair(piece_tokens, most_frequent_pair)
                for piece_tokens in tokens
            ]

            # 7. Add the new token to vocabulary
            self.vocab.add(most_frequent_pair[0] + most_frequent_pair[1])

            # 8. Save the merge rule
            self.merges.append(most_frequent_pair)

            print(f"Merge {i+1}: {most_frequent_pair}")

        # 9. Return all final pieces
        return tokens


# set(tokens) removes duplicate tokens.
# sorted(...) sorts the unique tokens.
# enumerate(...) assigns each token an integer ID.
# self.vocab stores the resulting dictionary.

    def build_vocab(self):

        unique_tokens = sorted(self.vocab)
        unique_tokens.append("<UNK>")

        self.vocab = {
            token: index
            for index, token in enumerate(unique_tokens)
        }
        self.itos={
            index:token
            for token,index in self.vocab.items()
        }
        self.vocab_size=len(self.vocab)

        return self.vocab

    
    def encode(self, text):

        # 1. Pre-tokenize the input text
        pieces = self.pre_tokenize(text)

        # 2. Process each piece separately
        final_tokens = []

        for piece in pieces:

            tokens = list(piece)

            # 3. Apply all learned merge rules
            for pair in self.merges:

                tokens = self.merge_pair(tokens, pair)

            # 4. Add the final tokens to our list
            final_tokens.extend(tokens)

        # 5. Convert tokens into token IDs
        token_ids = [
            self.vocab.get(token, self.vocab["<UNK>"])
            for token in final_tokens
        ]

        return token_ids


    def decode(self, token_ids):

        tokens = [
            self.itos[token_id]
            for token_id in token_ids
        ]

        return ''.join(tokens)

# data combines the vocabulary and merge rules.
# open(..., "w") creates or overwrites a JSON file.
# json.dump() writes Python data into JSON format.
# indent=4 makes the file readable.

    def save(self, file_path):

        data = {
            "vocab": self.vocab,
            "merges": self.merges
        }

        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)

        print("Tokenizer saved successfully!")

    def load(self, file_path):

        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        self.vocab = data["vocab"]

        self.itos = {
            index: token
            for token, index in self.vocab.items()
        }
        self.vocab_size=len(self.vocab)

        self.merges = [
            tuple(pair)
            for pair in data["merges"]
        ]

        print("Tokenizer loaded successfully!")

    def pre_tokenize(self, text):
        pieces = re.findall(
            r"\w+|[^\w\s]|\s+",
            text
        )
        return pieces

if __name__=="__main__":
    tokenizer = BPETokenizer()

    text = "banana apple"

    final_tokens = tokenizer.train(
        text,
        num_merges=5
    )

    print("\nFinal tokens:", final_tokens)
    print("Learned merges:", tokenizer.merges)

    vocab = tokenizer.build_vocab()

    print("\nVocabulary:", vocab)

    encoded = tokenizer.encode("banana")

    print("\nEncoded IDs:", encoded)

    decoded = tokenizer.decode(encoded)

    print("\nDecoded text:", decoded)

    test_text = "banana apple"

    encoded = tokenizer.encode(test_text)

    decoded = tokenizer.decode(encoded)

    print("\nOriginal:", test_text)
    print("Encoded:", encoded)
    print("Decoded:", decoded)

    print("Round trip successful:", test_text == decoded)

    print("banana:", tokenizer.encode("banana"))
    print("ban:", tokenizer.encode("ban"))
    print("an:", tokenizer.encode("an"))


    project_root = Path(__file__).resolve().parents[1]

    save_path = project_root / "checkpoints" / "bpe_tokenizer.json"

    save_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer.save(save_path)


    loaded_tokenizer = BPETokenizer()

    loaded_tokenizer.load(save_path)

    encoded = loaded_tokenizer.encode("banana")
    decoded = loaded_tokenizer.decode(encoded)

    print("Encoded:", encoded)
    print("Decoded:", decoded)

    text = "hello world! hello AI."

    pieces = tokenizer.pre_tokenize(text)

    print("Pre-tokenized pieces:", pieces)
