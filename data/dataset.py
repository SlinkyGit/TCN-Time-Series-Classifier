import torch
from torch.utils.data import Dataset


class ReturnDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float() # (N, C, L)
        self.y = torch.from_numpy(y).long() # (N, )

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    

