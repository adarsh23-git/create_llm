from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tokenizer import CharacterTokenizer
from dataset import LanguageDataset
from model import MiniGPT

project_root=Path(__file__).resolve().parents[1]

text=(project_root/"data"/"tiny.txt").read_text(
    encoding="utf-8"
)

tokenizer=CharacterTokenizer(text)   

encoded_data=tokenizer.encode(text)       #Convert characters into token IDs

block_size=3

dataset=LanguageDataset(                    #Creates input-target pairs
    encoded_data,
    block_size
)

dataloader=DataLoader(                     #Groups examples into batches
    dataset,
    batch_size=2,
    shuffle=True
)

model=MiniGPT(
    vocab_size=tokenizer.vocab_size,
    block_size=block_size,
    embedding_dim=32
)

loss_function=nn.CrossEntropyLoss()

optimizer=torch.optim.AdamW(
    model.parameters(),
    lr=0.001
)

print("model created successfully")

print("vocabulary size :",tokenizer.vocab_size)
print("dataset size :",len(dataset))

x,y=next(iter(dataloader))

print("input shape :",x.shape)
print("output shape :",y.shape)