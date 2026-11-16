import torch
from pathlib import Path

from model import MiniGPT
from tokenizer import CharacterTokenizer

project_root=Path(__file__).resolve().parents[1]
checkpoint_path=project_root/"checkpoint"/"minigpt.pt"

checkpoint=torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True
)

tokenizer=CharacterTokenizer("".join(checkpoint["chars"]))

model=MiniGPT(
    vocab_size=checkpoint["vocab_size"],
    block_size=checkpoint["block_size"],
    embedding_dim=checkpoint["embedding_dim"]
)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

def generate(model,tokenizer,prompt,max_new_token=50):
    context=torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long
    )

    with torch.no_grad():
        for _ in range(max_new_token):

            context_input=context[:, -model.position_embedding.num_embeddings:]

            logits=model(context_input)

            logits=logits[:,-1,:]

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
    prompt="hello",
    max_new_token=50
)

print("generated text")
print(result)
