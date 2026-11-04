from stage2.tokenizer import CharacterTokenizer

text = "hello world welcome to the world of llm and tokenizer"

tokenizer = CharacterTokenizer(text)
encoded=tokenizer.encode(text)
decoded=tokenizer.decode(encoded)

print(f"vocabulary :{tokenizer.chars}")
print(f"vocab size :{tokenizer.vocab_size}")
print(f"encoded :{encoded}")
print(f"decoded :{decoded}")
print(f"match :{decoded==text}")
