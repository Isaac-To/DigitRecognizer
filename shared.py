import torch
import torchvision.transforms as transforms
from torchvision.transforms import v2
import pandas as pd
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

def load_data(file_path, batch_size=1, shuffle=False, train=False) -> torch.utils.data.DataLoader:
    ds = pd.read_csv(file_path)
    # Check if the dataset has the 'label' column
    if 'label' not in ds.columns:
        label = np.zeros(len(ds), dtype=np.int64)
        data = ds.values
    else:
        label = ds['label'].values
        data = ds.drop(columns=['label']).values
    data = data.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0

    # If train create more data
    if train:
        transform = transforms.Compose([
            v2.RandomRotation(20),
            v2.RandomPerspective(distortion_scale=0.5, p=0.5),
            v2.RandomCrop((28, 28), padding=4),
        ])
        # Create more data
        data = np.concatenate([data, transform(torch.from_numpy(data)), transform(torch.from_numpy(data)), transform(torch.from_numpy(data)), transform(torch.from_numpy(data))], axis=0)
        label = np.concatenate([label, label, label, label, label], axis=0)

    # Convert to PyTorch tensors
    label = label.astype(np.int64)

    # Create a DataLoader
    ds = torch.utils.data.TensorDataset(torch.from_numpy(data), torch.from_numpy(label))
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return dl

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 64, (5, 5)),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(64, 128, (5, 5)),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.GELU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.seq(x)
        x = x.view(x.size(0), -1)
        x = F.log_softmax(x, dim=1)
        return x