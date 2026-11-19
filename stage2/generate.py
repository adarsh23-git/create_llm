import torch
from pathlib import Path

from model import MiniGPT
from tokenizer import CharacterTokenizer

#__file__ → location of the current Python file (generate.py).
# .resolve() → obtains its absolute path.
# .parents[1] → goes up two directory levels in the path's parent hierarchy.

project_root=Path(__file__).resolve().parents[1]
checkpoint_path=project_root/"checkpoints"/"minigpt.pt"

checkpoint=torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True
)
print("Checkpoint loaded successfully!")
print("Block size:", checkpoint["block_size"])
print("Embedding dimension:", checkpoint["embedding_dim"])
print("Vocabulary size:", checkpoint["vocab_size"])

tokenizer=CharacterTokenizer("".join(checkpoint["chars"]))

# recreate model Because minigpt.pt stores trained parameters, not the Python class definition itself.
# We first create the architecture, then load its trained parameters.

model=MiniGPT(
    vocab_size=checkpoint["vocab_size"],
    block_size=checkpoint["block_size"],
    embedding_dim=checkpoint["embedding_dim"]
)

#MiniGPT() → creates an empty model architecture with newly initialized parameters.
#load_state_dict() → replaces those parameters with the learned parameters from training.

model.load_state_dict(checkpoint["model_state_dict"])
print("Model weights loaded successfully!")

model.eval()

def generate(model,tokenizer,prompt,max_new_token=50,temperature=1.0):
    if temperature<=0:
        raise ValueError("temperature must be greater than 0")\
        
    context=torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long
    )

    with torch.no_grad():
        #The underscore _ means we don't need the loop counter.
        for _ in range(max_new_token):

            #model.position_embedding.num_embeddings retrieves the number of available positional embeddings,
            context_input=context[:, -model.position_embedding.num_embeddings:]

            #Get model predictions
            #The model processes the current context and returns vocabulary scores.

            logits=model(context_input)

            logits=logits[:,-1,:]

            # purpose of temperature is to control randomness
            # and purpose of Top_k is to restricts the number of candidate tokens

            logits=logits/temperature
            top_k=5
            values,indices=torch.topk(logits,top_k,dim=-1)
            filtered_logits=torch.full_like(logits,float('-inf'))
            filtered_logits.scatter_(-1,indices,values)
            probabilities=torch.softmax(logits,dim=-1)

            #argmax() — Always chooses the highest probability
            #multinomial()- To randomly sample a character according to its predicted probabilities, not stick with highest probability
            
            next_token=torch.multinomial(
                probabilities,
                num_samples=1
            )

            context=torch.cat(
                (context,next_token),
                dim=1
            )

        return tokenizer.decode(context[0].tolist())
result=generate(
    model,
    tokenizer,
    prompt="First citizen",
    max_new_token=50,
    temperature=1.0
)

print("generated text")
print(result)
