from stage2.tokenizer import CharacterTokenizer
from stage2.dataset import LanguageDataset
from torch.utils.data import DataLoader

text="hello"

tokenizer=CharacterTokenizer(text)
data=tokenizer.encode(text)
dataset=LanguageDataset(data,block_size=3)

print("total examples:",len(dataset))

x,y=dataset[0]
print("x:",x)
print("y:",y)

print("input text :",tokenizer.decode(x.tolist()))
print("target text:",tokenizer.decode(y.tolist()))

loader=DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

for x,y in loader:
    print("shape of x:", x.shape)
    print("shape of y:",y.shape)
    print("x:",x)
    print("y:",y)
    break


