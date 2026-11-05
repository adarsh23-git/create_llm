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