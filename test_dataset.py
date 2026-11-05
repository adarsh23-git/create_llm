from stage2.tokenizer import CharacterTokenizer
from stage2.dataset import LanguageDataset

text="hello world welcome to the world of llm and tokenizer"

tokenizer=CharacterTokenizer(text)
data=tokenizer.encode(text)
dataset=LanguageDataset(data,block_size=3)

print("total examples:",len(dataset))

x,y=dataset[0]
print("x:",x)
print("y:",y)

print("input text :",tokenizer.decode(x.tolist()))
print("target text:",tokenizer.decode(y.tolist()))