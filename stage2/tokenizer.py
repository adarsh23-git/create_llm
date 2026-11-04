class CharacterTokenizer:
    def __init__(self,text):
        # find unique characters in the text and sort them
        self.chars=sorted(list(set(text)))
        # vocabulary size is the number of unique characters
        self.vocab_size=len(self.chars)
        # Character -> Integer
        self.stoi={ch:i for i,ch in enumerate(self.chars)}
        # Integer -> Character
        self.itos={i:ch for i,ch in enumerate(self.chars)}

    def encode(self,text):
        return [self.stoi[ch] for ch in text]

    def decode(self,tokens):
        return ''.join(self.itos[i] for i in tokens)