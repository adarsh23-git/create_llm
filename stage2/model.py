import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniGPT(nn.Module):
    def __init__(self,vocab_size,block_size,embedding_dim):
        super().__init__()
        self.token_embedding=nn.Embedding(       # Converts each token ID into a learnable vector.
            vocab_size,
            embedding_dim
        )

        self.position_embedding=nn.Embedding(    # Teaches the model where each token occurs in the sequence.
            block_size,
            embedding_dim
        )


    def forward(self,x):
        batch_size,sequence_length=x.shape
        token_embedded=self.token_embedding(x)
        positions=torch.arange(
           sequence_length,
            device=x.device
        )

        position_embedded=self.position_embedding(positions)

        embedded=token_embedded+position_embedded
        return embedded