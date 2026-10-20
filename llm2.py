import torch.nn as nn
import torch
import torch.nn.functional as F



class ContextModel(nn.Module):
    def __init__(self,vocab_size,block_size,embedding_dim):
        super().__init__()

        self.embedding=nn.Embedding(vocab_size,embedding_dim)
        self.fc=nn.Linear(block_size*embedding_dim,vocab_size)

    def forward(self,x):
        embedded=self.embedding(x)
        combined=embedded.view(embedded.size(0),-1)
        logits=self.fc(combined)
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


# print(encode("hello i am adarsh"))

itos={i:ch for ch,i in stoi.items()}
print(itos) 

def decode(ids):
    return ''.join(itos[i] for i in ids)

# print(decode([8, 5, 10, 10, 13, 0, 9, 0, 1, 11, 0, 1, 4, 1, 14, 15, 8]))


encoded_text=encode(text)
decoded_text=decode(encoded_text)


print(f"encoded :{encoded_text}")
print(f"decoded :{decoded_text}")
print(f"match :{decoded_text==text}")

data=torch.tensor(encode(text),dtype=torch.long)
# print(data)
# print(data.shape)
# batch_size=3
# x=data[:batch_size].unsqueeze(0)
# y=data[batch_size].unsqueeze(0)




block_size=3

embedding_dim=8

model=ContextModel(
    vocab_size=len(char),
    block_size=block_size,
    embedding_dim=embedding_dim

)

# Query (Q): what a position is looking for.
# Key (K): what a position offers for matching.
# Value (V): the information a position contributes when attended to.


x=data[:block_size].unsqueeze(0)

embedded=model.embedding(x)

query_layer=nn.Linear(embedding_dim,embedding_dim,bias=False)
key_layer=nn.Linear(embedding_dim,embedding_dim,bias=False)

queries=query_layer(embedded)
keys=key_layer(embedded)

scores=torch.matmul(
    queries,
    keys.transpose(-2,-1)
)
mask=torch.tril(
    torch.ones(block_size,block_size)          # make lower traingular matrix
).bool()

scores=scores.masked_fill(
    ~mask,
    float("-inf")              # ~ is the logical operator ~mask represent true values and then true values is filled with negative infinity if it is simply mask then it target False values
)


weights=F.softmax(
    scores/(embedding_dim**0.5),
    dim=-1
)

print("causal mask")
print(mask)
print("embedded shape :",embedded.shape)
print("score shape :",scores.shape)
print("attention weight")
print(weights[0])

value_layer=nn.Linear(embedding_dim,embedding_dim,bias=False)
values=value_layer(embedded)
attended=torch.matmul(                     # matmul is used for matrix multiplication and helps to calculate attention scores
    weights, 
    values
)

print("value shape :",values.shape)
print("attention output shape :",attended.shape)
print("attended output :",attended)

# print("output shape :",output.shape)

# print("target id :",y)


# print("loss :",loss.item())

optimizer=torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

for epoch in range(1001):
    for i in range(len(data)-block_size):
        x=data[i:i+block_size].unsqueeze(0)
        y=data[i+block_size].unsqueeze(0)
    
        output=model(x)

        loss=F.cross_entropy(output,y)


        optimizer.zero_grad()

        loss.backward()
        optimizer.step()


    if epoch%100==0:
        print(f"epoch :{epoch}, loss :{loss.item()}")





# model.eval()

# with torch.no_grad():
#     for i in range(5):
#         x=data[i:i+block_size].unsqueeze(0)




#         output=model(x)

#         predicted_id=output.argmax(dim=1).item()

#         actual_id=data[i+block_size].item()

#         print(f"input :{decode(x[0].tolist())!r}")
#         print(f"predicted :{itos[predicted_id]!r}")
#         print(f"Actual :{itos[actual_id]!r}")


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








