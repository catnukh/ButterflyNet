import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import random_split
import os
from PIL import Image

train_set = pd.read_csv("Training_set.csv")
test_set = pd.read_csv("Testing_set.csv")
classes = train_set.label.unique()

class ButterfliesDataset(Dataset):
    def __init__(self, df, path, transform=None):
        self.df = df
        self.path = path
        self.transform = transform
        self.classes = sorted(classes)
        self.classes_index = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.path, self.df.iloc[idx]['filename'])
        image = Image.open(img_path)
        label = self.classes_index[self.df.iloc[idx]['label']]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((150, 150)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((150, 150)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_img_df = ButterfliesDataset(train_set, "train", transform_train)
val_img_df = ButterfliesDataset(train_set, "train", transform_test)
test_img_df = ButterfliesDataset(test_set, "test", transform_test)

train_size = int(len(train_img_df) * 0.8)
val_size = len(train_img_df) - train_size
train_indices, val_indices = random_split(range(len(train_img_df)), [train_size, val_size])

train_subset = torch.utils.data.Subset(train_img_df, train_indices)
val_subset = torch.utils.data.Subset(val_img_df, val_indices)

train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_subset, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_img_df, batch_size=64, shuffle=False)