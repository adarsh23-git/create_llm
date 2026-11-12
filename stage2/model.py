import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):

    def __init__(self, embedding_dim, block_size):

        super().__init__()

        self.query_layer = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key_layer = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value_layer = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.block_size = block_size
        self.embedding_dim = embedding_dim

        self.num_heads=2
        self.head_dim=embedding_dim // self.num_heads

        self.output_layer=nn.Linear(
            embedding_dim,
            embedding_dim
        )

    def forward(self, x):

        batch_size, sequence_length, embedding_dim = x.shape

        queries = self.query_layer(x)
        keys = self.key_layer(x)
        values = self.value_layer(x)

        queries=queries.view(                 #view() is used to reshape a tensor without changing its actual data values.
            batch_size,                       #We can convert a 1D tensor into a 2D tensor using view()
            sequence_length,                       
            self.num_heads,
            self.head_dim
        )
        keys = keys.view(                    #[2, 3, 2, 4]
            batch_size,                          #Batch × Sequence × Heads × Head dimension
            sequence_length,
            self.num_heads,
            self.head_dim
        )
 
        values = values.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
    
        scores = torch.matmul(
            queries,
            keys.transpose(-2, -1)
        )

        scores = scores / (self.head_dim ** 0.5)

        mask = torch.tril(     #torch.ones() creates a matrix filled with ones.
            torch.ones(        #torch.tril() keeps the lower triangle.
                sequence_length,
                sequence_length,
                device=x.device
            )
        ).bool()             #.bool() converts values into True/False.

        scores = scores.masked_fill(
            ~mask,                        #~mask reverses True and False.
            float('-inf')                  #masked_fill() replaces future-token scores with negative infinity
        )

        attention_weights = F.softmax(scores, dim=-1) # dim=-1 means row-wise softmax

        attention_output = torch.matmul(          # attention(Q,K,V)=softmax(QK^T)V
            attention_weights,
            values
        )
        attention_output = attention_output.transpose(1, 2)

        attention_output=attention_output.contiguous().view(   #transpose() changes how PyTorch views the tensor in memory without necessarily rearranging the underlying data. .contiguous() creates a contiguous memory layout so that .view() can safely reshape it.
            batch_size,
            sequence_length,
            self.embedding_dim
        )

        attention_output=self.output_layer(attention_output)

        return attention_output
    
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

        # Create attention only once
        self.attention = CausalSelfAttention(
            embedding_dim,
            block_size
        )


    def forward(self,x):
        batch_size,sequence_length=x.shape
        token_embedded=self.token_embedding(x)
        positions=torch.arange(
           sequence_length,
           device=x.device
        )

        position_embedded=self.position_embedding(positions)

        # Combine token and positional embeddings
        embedded = token_embedded + position_embedded

        attention_output=self.attention(embedded)
        return attention_output


#     The output shape is [4, 5, 16].


# -   4 = batch size (4 examples).
# -   5 = block size (5 tokens per example).
# -   16 = embedding dimension (each token gets a vector of 16 numbers).