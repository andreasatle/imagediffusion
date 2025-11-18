# datasets/mnist_manager.py

import math
import torch
from torch.utils.data import random_split
from torchvision import datasets, transforms

NULL_CLASS = 10


class ScaleToMinus1To1:
    def __call__(self, x):
        return x * 2 - 1


class MNISTDatasetManager:
    """
    Clean wrapper around MNIST:
      - normalization
      - train/val split
      - batch sampling
      - label drop (CFG)
      - conditional label tiling
    """

    def __init__(self, root="./data", val_fraction=0.1, device="cpu"):
        self.device = device

        transform = transforms.Compose([
            transforms.ToTensor(),
            ScaleToMinus1To1(),
        ])

        full_ds = datasets.MNIST(
            root=root,
            train=True,
            download=True,
            transform=transform,
        )

        val_size = int(len(full_ds) * val_fraction)
        train_size = len(full_ds) - val_size
        self.train_ds, self.val_ds = random_split(full_ds, [train_size, val_size])

    # -----------------------------------------------------

    def sample_batch(self, dataset, batch_size, drop_prob=0.0):
        """
        Draw a random batch from train or val set.
        Handles class dropping for classifier-free guidance.
        """
        idx = torch.randint(0, len(dataset), (batch_size,))
        batch = [dataset[i] for i in idx.tolist()]

        x = torch.stack([item[0] for item in batch]).to(self.device)
        labels = torch.tensor([item[1] for item in batch], device=self.device)

        if drop_prob > 0.0:
            mask = torch.rand(batch_size, device=self.device) < drop_prob
            labels = torch.where(mask, torch.full_like(labels, NULL_CLASS), labels)

        return x, labels

    # -----------------------------------------------------

    def train_batch(self, batch_size, drop_prob=0.0):
        return self.sample_batch(self.train_ds, batch_size, drop_prob)

    def val_batch(self, batch_size):
        return self.sample_batch(self.val_ds, batch_size, drop_prob=0.0)

    # -----------------------------------------------------

    def tiled_labels(self, label_list, batch_size):
        """
        For generating class-conditioned sampling grids.
        """
        if not label_list:
            return None

        tiled = (label_list * math.ceil(batch_size / len(label_list)))[:batch_size]
        return torch.tensor(tiled, device=self.device, dtype=torch.long)
