from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Optional, Callable


class MyDataset(Dataset):
    def __init__(self, split: str = "train", transform: Optional[Callable] = None):
        available_splits = ["train", "validation", "test"]
        assert any(
            split.startswith(s) for s in available_splits
        ), f"Invalid split name '{split}'. Use one of {available_splits}."

        self.split = split
        self.transform = transform

        try:
            self.dataset = load_dataset("Last-Bullet/LaTeX_Spoken", split=split)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load dataset 'Last-Bullet/LaTeX_Spoken' with split '{split}': {e}"
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]

        if self.transform:
            item = self.transform(item)

        return item
