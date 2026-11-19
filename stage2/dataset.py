import torch
import torch.utils.data as Dataset

class LanguageDataset(Dataset.Dataset):
    def __init__(self,data,block_size):
        self.data=torch.tensor(data,dtype=torch.long)
        self.block_size=block_size

    def __len__(self):
        return len(self.data)-self.block_size

    def __getitem__(self,index):
        x=self.data[index:index+self.block_size]
        y=self.data[index+1:index+self.block_size+1]
        return x,y

# dataset bada ho gaya isliye kiya

#get_random_batch() randomly selects starting positions in the text
def get_random_batch(data,batch_size,block_size):
    max_index=len(data)- block_size-1
    indices=torch.randint(
        0,
        max_index+1,
        (batch_size,)
    )

    x=torch.stack([
        data[i:i+block_size] for i in indices
    ])

    y=torch.stack([
        data[i+1:i+block_size+1] for i in indices
    ])

    return x,y