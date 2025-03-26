import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import ast
import os
from torchvision import transforms
from torch.utils.data import DataLoader

class ShapeDataset(Dataset):
    def __init__(self, csv_file, dataset_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.dataset_dir = dataset_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_rel_path = self.data.iloc[idx]['image_path']
        img_path = os.path.join(self.dataset_dir, img_rel_path)
        
        image = Image.open(img_path).convert("RGB")

        label = ast.literal_eval(self.data.iloc[idx]['label'])

        if self.transform:
            image = self.transform(image)

        return image, label

    

if __name__ == '__main__':


    # Set paths
    csv_file = "/home/kiran/projects/github/test/all-shapes-and-colors/train.csv"
    dataset_dir = "/home/kiran/projects/github/test/all-shapes-and-colors/dataset"

    # Define transform
    transform = transforms.ToTensor()

    # Create dataset
    dataset = ShapeDataset(csv_file=csv_file, dataset_dir=dataset_dir, transform=transform)

    # Optional: create a DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Test loading a few samples
    for i, (image, label) in enumerate(dataloader):
        print(f"Image shape: {image.shape}")
        print(f"Label: {len(label)}")

        if len(label) > 0:
            for j in range(len(label)):
                print(f"Label[{j}]: {label[j][0]} {label[j][1]}")

        if i == 2:
            break
