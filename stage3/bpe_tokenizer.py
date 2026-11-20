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



tokenizer = BPETokenizer()

tokens = ["a", "b", "c", "a", "b"]

pairs = tokenizer.get_pairs(tokens)

print(pairs)

pair_frequency = tokenizer.count_pairs(pairs)

print(pair_frequency)

most_frequent_pair = tokenizer.get_most_frequent_pair(
    pair_frequency
)

print("Most frequent pair:", most_frequent_pair)

new_tokens = tokenizer.merge_pair(
    tokens,
    most_frequent_pair
)

print("Before merging:", tokens)
print("After merging:", new_tokens)