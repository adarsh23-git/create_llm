import torch.nn as nn
import torch
import torch.nn.functional as F



class ContextModel(nn.Module):
    def __init__(self,vocab_size,block_size,embedding_dim):
        super().__init__()

        self.num_heads=2

        self.embedding=nn.Embedding(vocab_size,embedding_dim)
        self.fc=nn.Linear(block_size*embedding_dim,vocab_size)

        self.head_dim=embedding_dim // self.num_heads

        self.query_layer=nn.Linear(embedding_dim,embedding_dim,bias=False)    #query_layer learns how each position searches for relevant information.
        self.key_layer=nn.Linear(embedding_dim,embedding_dim,bias=False)      #key_layer learns how each position is matched against queries.
        self.value_layer=nn.Linear(embedding_dim,embedding_dim,bias=False)    #value_layer learns the information each position contributes.

        self.postion_embedding=nn.Embedding(block_size,embedding_dim)
        self.ffn=nn.Sequential(
            nn.Linear(embedding_dim,embedding_dim*4),
            nn.ReLU(),
            nn.Linear(embedding_dim*4,embedding_dim)
        )

        self.ln1=nn.LayerNorm(embedding_dim)
        self.ln2=nn.LayerNorm(embedding_dim)

    def forward(self,x):
        embedded=self.embedding(x)

        position=torch.arange(x.size(1),device=x.device) # arrange create postion Id for context 
        position_embedded=self.postion_embedding(position)   # convert Id`s in learnable vectors`

        embedded=embedded+position_embedded       # combine character identity and position information

        queries=self.query_layer(embedded)
        
        queries=queries.view(                         # It reshapes the tensor without changing its underlying data.
            queries.size(0),
            queries.size(1),
            self.num_heads,
            self.head_dim      
        )
        keys=self.key_layer(embedded)
        values=self.value_layer(embedded)

        scores=torch.matmul(
            queries,
            keys.transpose(-2,-1)
        )

        mask=torch.tril(
            torch.ones(
                x.size(1),
                x.size(1),
                device=x.device
            )
        ).bool()

        scores=scores.masked_fill(
            ~mask,
            float("-inf")
        )

        weights=F.softmax(
            scores/(queries.size(-1)**0.5),
            dim=-1
        )

        attended=torch.matmul(weights,values)
        x=self.ln1(embedded+attended)
        ffn_output=self.ffn(attended)
        x=self.ln2(x+ffn_output)

        # combined = attended.reshape(attended.size(0), -1)
        combined=x.reshape(x.size(0),-1)
        logits = self.fc(combined)

        return logits


text="hello i am adarsh and i am learning to build llm from scratch"

print(text)
print(len(text))
char=sorted(set(text))
print(char)
print(len(char))

stoi={ch:i for i,ch in enumerate(char)}
print(stoi)

def encode(text):
    return [stoi[ch] for ch in text]


itos={i:ch for ch,i in stoi.items()}
print(itos) 

def decode(ids):
    return ''.join(itos[i] for i in ids)



encoded_text=encode(text)
decoded_text=decode(encoded_text)


print(f"encoded :{encoded_text}")
print(f"decoded :{decoded_text}")
print(f"match :{decoded_text==text}")

data=torch.tensor(encode(text),dtype=torch.long)

block_size=3

embedding_dim=8

model=ContextModel(
    vocab_size=len(char),
    block_size=block_size,
    embedding_dim=embedding_dim

)


optimizer=torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

for epoch in range(1001):
    totalloss=0
    for i in range(len(data)-block_size):
        x=data[i:i+block_size].unsqueeze(0)
        y=data[i+block_size].unsqueeze(0)
    
        output=model(x)

        loss=F.cross_entropy(output,y)


        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        totalloss+=loss.item()


    if epoch%100==0:
        avgloss=totalloss/(len(data)-block_size)
        print(f"epoch :{epoch}, loss :{avgloss:.2f}")


model.eval()

correct=0
total=0

with torch.no_grad():
    for i in range(len(data)-block_size):
        x=data[i:i+block_size].unsqueeze(0)
        y=data[i+block_size].unsqueeze(0)

        logits=model(x)
        predicted_id=logits.argmax(dim=1).item()
        actual_id=y.item()

        if predicted_id ==actual_id:
            correct+=1

        total+=1

acurracy=correct/total*100

print("accuracy :",acurracy)



def generate(start_text, max_new_chars=20):
    model.eval()

    context = torch.tensor(
        encode(start_text),
        dtype=torch.long
    ).unsqueeze(0)

    generated = start_text

    with torch.no_grad():
        for _ in range(max_new_chars):
            # Keep exactly the last block_size characters
            context = context[:, -block_size:]

            logits = model(context)

            probabilities = F.softmax(logits / 0.8, dim=1)
            next_id = torch.multinomial(
                probabilities, num_samples=1
            ).item()

            generated += itos[next_id]

            next_char = torch.tensor([[next_id]], dtype=torch.long)
            context = torch.cat((context, next_char), dim=1)

    return generated




print(generate("hello i am ad",20))