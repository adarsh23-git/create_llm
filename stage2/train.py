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

block_size=32

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

num_epochs=100

for epoch in range(num_epochs):
    total_loss=0
    for x,y in dataloader:
        optimizer.zero_grad()

        logits=model(x)         #	Predicts next-token logits

        batch_size,sequence_length,vocab_size=logits.shape

        loss=loss_function(
            logits.reshape(batch_size*sequence_length,vocab_size),
            y.reshape(batch_size*sequence_length)
        )

        loss.backward()

        optimizer.step()

        total_loss+=loss.item()

    average_loss=total_loss/len(dataloader)

    if (epoch+1)%10==0:
        print(f"epoch[{epoch+1}/{num_epochs}], loss :{average_loss:.4f}")

print("vocabulary size :",tokenizer.vocab_size)
print("dataset size :",len(dataset))

x,y=next(iter(dataloader))

print("input shape :",x.shape)
print("output shape :",y.shape)


checkpoint_path=project_root/"checkpoint"/"minigpt.pt"
checkpoint_path.parent.mkdir(parents=True,exist_ok=True)

torch.save({
    "model_state_dict":model.state_dict(),
    "vocab_size":tokenizer.vocab_size,
    "block_size":block_size,
    "embedding_dim":32,
    "chars":tokenizer.chars
},checkpoint_path)

print("model saved successfully")