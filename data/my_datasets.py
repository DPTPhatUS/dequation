import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

class MathBridge(Dataset):
    def __init__(self, split: str = 'train'):
        assert split.startswith('train') or split.startswith('validation') or split.startswith('test'), "Invalid split name. Use 'train', 'validation', or 'test'."

        self.split = split
        self.dataset = load_dataset('Last-Bullet/MathBridge_Splitted', split=split)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]