import os
import pandas as pd
from torch.utils.data import Dataset
import torch
from torchvision.io import read_image
import torchvision.transforms
import pickle
EAI_TRANSFORM = torchvision.transforms.Compose([
     torchvision.transforms.Resize((224, 224)),
     torchvision.transforms.ConvertImageDtype(torch.float32),
])

class UnsupervisedImageDataset(Dataset):
    def __init__(self, embed_file, transform=None):
        with open(embed_file, "rb") as fo:
            embed = pickle.load(fo)
        self.img_files = embed["files"]
        self.embedded_data = embed["embeddings"]
        
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # TODO: Check if iterable or idx
        try:
            images = torch.stack([read_image(self.img_files[i]) for i in idx])
            print(images.shape)
        except TypeError:
            images = read_image(self.img_files[idx])

        if self.transform is not None:
            images = self.transform(images)
        return images, self.embedded_data[idx]

if __name__ == "__main__":
    EMBED_PATH="embeddings.pkl"
    dataset = UnsupervisedImageDataset(EMBED_PATH, transform=EAI_TRANSFORM)
    print(f"Dataset length: {len(dataset)}")
    import numpy as np
    for i in range(len(dataset)):
        img, _ = dataset[i]
        print(img.shape)
        print(torch.max(img))
        print(torch.min(img))

