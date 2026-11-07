import torch
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

output=model(x)

print("input shape :",x.shape)
print("output shape :",output.shape)