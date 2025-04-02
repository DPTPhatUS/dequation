import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

class MathBridge(Dataset):
    def __init__(self, phase: str | None = 'train'):
        assert phase in ['train', 'test', 'validation'], f"Invalid phase: {phase}. Choose from 'train', 'test', or 'validation'."

        super(MathBridge, self).__init__()
        self.dataset = load_dataset('Last-Bullet/MathBridge_Splitted', split=phase)
        self.phase = phase

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]