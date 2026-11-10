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

        self.query_layer=nn.Linear(embedding_dim,embedding_dim,bias=False)
        self.key_layer=nn.Linear(embedding_dim,embedding_dim,bias=False)
        self.value_layer=nn.Linear(embedding_dim,embedding_dim,bias=False)


    def forward(self,x):
        batch_size,sequence_length=x.shape
        token_embedded=self.token_embedding(x)
        positions=torch.arange(
           sequence_length,
           device=x.device
        )

        position_embedded=self.position_embedding(positions)

        embedded=token_embedded+position_embedded
        queries=self.query_layer(embedded)
        keys=self.key_layer(embedded)
        values=self.value_layer(embedded)
        print("queries shape :",queries.shape)
        print("key shape :",keys.shape)
        print("values shape :",values.shape)

        scores=torch.matmul(                    # attention score = Q x K^T   -1 is last dimension and -2 next last
            queries,
            keys.transpose(-2,-1)
        )
        print("scores shape :",scores.shape)
        print("scores :",scores)

        mask=torch.tril(                #torch.ones() creates a matrix filled with ones.
        torch.ones(                      #torch.tril() keeps the lower triangle.
            sequence_length,
            sequence_length,
            device=x.device
            )
        ).bool()                         #.bool() converts values into True/False.

        scores=scores.masked_fill(
            ~mask,                       #~mask reverses True and False.
            float('-inf')                #masked_fill() replaces future-token scores with negative infinity.
        )

        attention_weights=F.softmax(scores,dim=-1)     # dim=-1 means row-wise softmax
        attention_output=torch.matmul(attention_weights,values)       # attention(Q,K,V)=softmax(QK^T)V
        print("attention weight shape :",attention_weights.shape)
        print("attention weight :",attention_weights)
        print("attention output shape :",attention_output.shape)


        return embedded



#     The output shape is [4, 5, 16].


# -   4 = batch size (4 examples).
# -   5 = block size (5 tokens per example).
# -   16 = embedding dimension (each token gets a vector of 16 numbers).