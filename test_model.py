import torch
import torch.nn as nn
from stage2.model import MiniGPT

model=MiniGPT(
    vocab_size=17,
    block_size=3,
    embedding_dim=8
)

x=torch.tensor([
    [1,2,3],
    [4,5,6]
])

y=torch.tensor([
    [2,3,4],
    [5,6,7]
])

logits=model(x)

print("input shape :",x.shape)
print("output shape :",logits.shape)   

loss_function=nn.CrossEntropyLoss()
batch_size,sequence_length,vocab_size=logits.shape

logits=logits.view(                            #.view() reshapes predictions from [2, 3, 17] to [6, 17]
    batch_size*sequence_length,
    vocab_size
)

targets=y.view(                                #Targets are reshaped from [2, 3] to [6].
    batch_size*sequence_length
)

loss=loss_function(logits,targets)

print("loss :",loss.item())
