import torch
import pandas as pd
from PIL import Image
import os


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, images_path: str, labels_path: str, transform=None):
        super().__init__()
        self.images_path = images_path
        self.labels_df = pd.read_csv(labels_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels_df)

    def __getitem__(self, idx: int):
        suite_id, sample_id, code, value, character = self.labels_df.loc[idx, :]
        path = os.path.join(self.images_path, f"input_{suite_id}_{sample_id}_{code}.jpg")
        sample = Image.open(path)
        if self.transform:
            sample = self.transform(sample)

        return sample, code-1 # code goes from 1 to 15