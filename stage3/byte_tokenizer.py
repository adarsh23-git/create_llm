
import json
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
tokenizer_path = project_root / "checkpoints" / "byte_bpe_tokenizer.json"


class ByteBPETokenizer:

    def __init__(self):

        # Initial vocabulary contains 256 byte tokens
        self.vocab = {
            bytes([i]): i
            for i in range(256)
        }

        self.itos = {
            index: token
            for token, index in self.vocab.items()
        }

        self.merges = []

        self.vocab_size = len(self.vocab)

    # -------- BYTE ENCODING --------

    def encode_bytes(self, text):

        # Convert text into UTF-8 bytes
        byte_values = text.encode("utf-8")

        # Convert bytes into individual byte tokens
        tokens = [
            bytes([byte])
            for byte in byte_values
        ]

        return tokens

    # -------- BYTE DECODING --------

    def decode_bytes(self, tokens):

        # Combine all byte tokens
        byte_data = b"".join(tokens)

        # Convert UTF-8 bytes back into text
        text = byte_data.decode("utf-8")

        return text

    # -------- PAIR OPERATIONS --------

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

    # -------- BPE TRAINING --------

    def train(self, text, num_merges):

        # Reset vocabulary and merges before training
        self.vocab = {
            bytes([i]): i
            for i in range(256)
        }

        self.itos = {
            index: token
            for token, index in self.vocab.items()
        }

        self.merges = []

        # 1. Convert text into initial byte tokens
        tokens = self.encode_bytes(text)

        # 2. Repeat the merging process
        for i in range(num_merges):

            # 3. Get adjacent pairs
            pairs = self.get_pairs(tokens)

            # 4. Count pair frequencies
            pair_frequency = self.count_pairs(pairs)

            # 5. Find the most frequent pair
            most_frequent_pair = self.get_most_frequent_pair(
                pair_frequency
            )

            # Stop if no pairs are available
            if most_frequent_pair is None:
                break

            # 6. Merge the most frequent pair
            tokens = self.merge_pair(
                tokens,
                most_frequent_pair
            )

            # 7. Create the new token
            new_token = (
                most_frequent_pair[0]
                + most_frequent_pair[1]
            )

            # 8. Add new token to vocabulary
            new_id = len(self.vocab)

            self.vocab[new_token] = new_id
            self.itos[new_id] = new_token

            # 9. Save the merge rule
            self.merges.append(most_frequent_pair)

            print(
                f"Merge {i+1}: {most_frequent_pair}"
                f" -> {new_token}"
            )

        # 10. Update vocabulary size
        self.vocab_size = len(self.vocab)

        return tokens

    # -------- TEXT ENCODING --------

    def encode(self, text):

        # 1. Convert text into initial byte tokens
        tokens = self.encode_bytes(text)

        # 2. Apply all learned merge rules in order
        for pair in self.merges:

            tokens = self.merge_pair(
                tokens,
                pair
            )

        # 3. Convert final tokens into token IDs
        token_ids = [
            self.vocab[token]
            for token in tokens
        ]

        return token_ids

    # -------- TOKEN ID DECODING --------

    def decode(self, token_ids):

        # 1. Convert token IDs back into byte tokens
        tokens = [
            self.itos[token_id]
            for token_id in token_ids
        ]

        # 2. Combine bytes and decode UTF-8
        text = self.decode_bytes(tokens)

        return text

    # -------- SAVE TOKENIZER --------

    def save(self, file_path):

        # Store merge rules as hexadecimal strings
        data = {
            "merges": [
                [
                    pair[0].hex(),
                    pair[1].hex()
                ]
                for pair in self.merges
            ]
        }

        with open(file_path, "w", encoding="utf-8") as file:

            json.dump(data, file, indent=4)

        print("Tokenizer saved successfully!")

    # -------- LOAD TOKENIZER --------

    def load(self, file_path):

        with open(file_path, "r", encoding="utf-8") as file:

            data = json.load(file)

        # Recreate initial byte vocabulary
        self.vocab = {
            bytes([i]): i
            for i in range(256)
        }

        self.itos = {
            index: token
            for token, index in self.vocab.items()
        }

        self.merges = []

        # Rebuild vocabulary using saved merge rules
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


# -------- TESTING --------

if __name__ == "__main__":

    tokenizer = ByteBPETokenizer()

    # Training text
    text = "banana banana"

    print("Original text:", text)

    # Train BPE
    tokenizer.train(
        text,
        num_merges=5
    )

    print("\nVocabulary size:", tokenizer.vocab_size)

    # Test encoding
    test_text = "banana"

    encoded = tokenizer.encode(test_text)

    print("\nTest text:", test_text)
    print("Encoded IDs:", encoded)

    # Test decoding
    decoded = tokenizer.decode(encoded)

    print("Decoded text:", decoded)

    # Test unseen Unicode character
    unicode_text = "Hello ₹"

    unicode_encoded = tokenizer.encode(unicode_text)
    unicode_decoded = tokenizer.decode(unicode_encoded)

    print("\nUnicode text:", unicode_text)
    print("Unicode encoded:", unicode_encoded)
    print("Unicode decoded:", unicode_decoded)

    # Save tokenizer
    tokenizer.save(tokenizer_path)

    # Load tokenizer into a new object
    loaded_tokenizer = ByteBPETokenizer()

    loaded_tokenizer.load(tokenizer_path)

    # Test loaded tokenizer
    loaded_encoded = loaded_tokenizer.encode(test_text)
    loaded_decoded = loaded_tokenizer.decode(loaded_encoded)

    print("\nLoaded tokenizer output:")
    print("Encoded IDs:", loaded_encoded)
    print("Decoded text:", loaded_decoded)

    # Final round-trip test
    print(
        "\nRound trip:",
        loaded_decoded == test_text
    )
