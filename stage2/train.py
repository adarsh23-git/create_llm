from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# from tokenizer import CharacterTokenizer
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from stage3.bpe_tokenizer import BPETokenizer
from dataset import LanguageDataset
from dataset import get_random_batch
from model import MiniGPT

project_root=Path(__file__).resolve().parents[1]

text=(project_root/"data"/"tiny.txt").read_text(
    encoding="utf-8"
)

# tokenizer=CharacterTokenizer(text)
tokenizer=BPETokenizer()

tokenizer_path=project_root/"checkpoints"/"bpe_tokenizer.json"
tokenizer.load(tokenizer_path)

encoded_data=tokenizer.encode(text)       #Convert characters into token IDs

# Convert encoded data into PyTorch tensor
data=torch.tensor(encoded_data,dtype=torch.long)

# split data into training and validation
split_index=int(0.9*len(data))

train_data=data[:split_index]
val_data=data[split_index:]

print("total tokens :",len(data))
print("training tokens :",len(train_data))
print("validation tokens :",len(val_data))

# Configuration
block_size=64
batch_size=16
embedding_dim=64

max_steps=5000
eval_interval=100
eval_iters=20

# dataset=LanguageDataset(                    #Creates input-target pairs
#     encoded_data,
#     block_size
# )

# train_dataset=LanguageDataset(train_data,block_size)
# val_dataset=LanguageDataset(val_data,block_size)

# print("training dataset size :",len(train_dataset))
# print("validation dataset size :",len(val_dataset))

# dataloader=DataLoader(                     #Groups examples into batches
#     dataset,
#     batch_size=2,
#     shuffle=True
# )

# train_loader = DataLoader(
#     train_dataset,
#     batch_size=32,
#     shuffle=True
# )

# val_loader = DataLoader(
#     val_dataset,
#     batch_size=32,
#     shuffle=False
# )

model=MiniGPT(
    vocab_size=tokenizer.vocab_size,
    block_size=block_size,
    embedding_dim=embedding_dim
)

loss_function=nn.CrossEntropyLoss()


# T_max=max_steps: the learning rate follows its cosine decay over 5,000 steps.
# eta_min=0.0001: the minimum learning rate.
# learning_rate=0.001: the starting learning rate.

learning_rate=0.001
optimizer=torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)

scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=max_steps,
    eta_min=0.0001
)

print("model created successfully")


# -------- EVALUATION FUNCTION --------

def estimate_loss():

    model.eval()

    results={}

    with torch.no_grad():

        # Evaluate both training and validation data
        for split,split_data in [
            ("train",train_data),
            ("val",val_data)
        ]:

            losses=torch.zeros(eval_iters)

            for k in range(eval_iters):

                x,y=get_random_batch(
                    split_data,
                    batch_size,
                    block_size
                )

                logits=model(x)

                batch_size_current,sequence_length,vocab_size=logits.shape

                loss=loss_function(
                    logits.reshape(
                        batch_size_current*sequence_length,
                        vocab_size
                    ),
                    y.reshape(
                        batch_size_current*sequence_length
                    )
                )

                losses[k]=loss.item()

            results[split]=losses.mean().item()

    model.train()

    return results


# -------- TRAINING LOOP --------

for step in range(max_steps):

    # Get random training batch
    x,y=get_random_batch(
        train_data,
        batch_size,
        block_size
    )

    # Predict next-token logits
    logits=model(x)

    batch_size_current,sequence_length,vocab_size=logits.shape

    loss=loss_function(
        logits.reshape(
            batch_size_current*sequence_length,
            vocab_size
        ),
        y.reshape(
            batch_size_current*sequence_length
        )
    )

    # zero_grad() — clear previous gradients.
    # backward() — calculate new gradients.
    # optimizer.step() — update model parameters.
    # scheduler.step() — adjust the learning rate for the next update.

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    scheduler.step()

    # -------- VALIDATION --------

    if (step+1)%eval_interval==0:

        losses=estimate_loss()

	    # Reads current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Step [{step+1}/{max_steps}] | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {losses['train']:.4f} | "
            f"Validation Loss: {losses['val']:.4f}",
            flush=True
        )


# Test batch shapes
x,y=get_random_batch(
    train_data,
    batch_size,
    block_size
)

print("Vocabulary size:",tokenizer.vocab_size)
print("Training tokens:",len(train_data))
print("Input shape:",x.shape)
print("Target shape:",y.shape)


# -------- SAVE CHECKPOINT --------

checkpoint_path=project_root/"checkpoints"/"minigpt_bpe_v2.pt"

checkpoint_path.parent.mkdir(parents=True,exist_ok=True)

torch.save({
    "model_state_dict":model.state_dict(),
    "vocab_size":tokenizer.vocab_size,
    "block_size":block_size,
    "embedding_dim":embedding_dim,
    # "chars":tokenizer.chars
    "tokenizer_type":"BPE"
},checkpoint_path)

print("model saved successfully")
