text = "Hello ₹"

# Convert text into UTF-8 bytes
byte_values = list(text.encode("utf-8"))

print("Original text:", text)
print("Byte values:", byte_values)
print("Number of bytes:", len(byte_values))

# Create all possible byte values
byte_vocab = list(range(256))

print("Vocabulary size:", len(byte_vocab))
print("First 10 tokens:", byte_vocab[:10])
print("Last 10 tokens:", byte_vocab[-10:])