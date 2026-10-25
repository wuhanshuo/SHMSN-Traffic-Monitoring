# SPDX-License-Identifier: AGPL-3.0-only
"""
In-memory datasets of 5 s sensor windows.

Each training sample is a pickle file holding a dict (see tools/generator.py, tools/feat_write.py):

    YNLRegDataset4Cls  {"signal": float array (2, 500, 8),  "label": [LeftCar, RightCar, LeftTruck, RightTruck]}
    FeatDataset4Cls    {"feat":   float array (2, 8, 8),    "label": [LeftCar, RightCar, LeftTruck, RightTruck]}

Axis 0 is the modality (0 = acceleration, 1 = strain), the last axis the 8 sensor nodes.
Labels are fractional vehicle counts: a vehicle only partly inside the window counts with
the fraction of its passage that falls into the window.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
import random
from glob import glob
import pickle


def collate_fn(data):
    """
    Args:
        data: a list of (signal, label) tuples, len(data) = batch_size
    Returns:
        [tuple of signals, tuple of labels]
    """
    return list(zip(*data))


def add_gaussian_noise_standardized(data, mean=0, std=0.01):
    """
    Add Gaussian noise to standardized sensor data.

    Args:
        data: Standardized sensor data (PyTorch tensor or numpy array)
        mean: Mean of the Gaussian noise
        std: Standard deviation of the Gaussian noise (controls the intensity of the noise)

    Returns:
        Noisy data of the same type as `data`
    """
    if isinstance(data, torch.Tensor):
        noise = torch.normal(mean=mean, std=std, size=data.shape).to(data.device)
        return data + noise
    return data + np.random.normal(0, std, data.shape)


class YNLRegDataset4Cls(Dataset):
    """
    Raw-signal windows for the end-to-end models.

    Args:
        dataset_dir: folder with *.pkl samples
        channel: modalities to keep, [0] = acceleration, [1] = strain, [0, 1] = both
        augment: add Gaussian noise (std = `noise`) to 80 % of the samples when read
        noise: standard deviation of the augmentation noise
    """

    def __init__(self, dataset_dir, channel=[0], augment=False, noise=0.02, LOGGER=None):
        self.dataset_dir = dataset_dir
        self.channel = channel
        self.augment = augment
        self.noise = noise

        if LOGGER is not None:
            LOGGER.info(f"Noise std is {self.noise}")

        self.data = []
        self.labels = []
        self.files = glob(f"{self.dataset_dir}/*.pkl")

        if len(self.files) == 0:
            raise ValueError(f"No data is found in {self.dataset_dir}!")

        # Preload everything into memory
        for item_path in self.files:
            with open(item_path, "rb") as f:
                item_dict = pickle.load(f)
                signal = item_dict["signal"][self.channel]  # (len(channel), 500, 8)
                label = item_dict["label"]

                self.data.append(torch.tensor(signal.copy(), dtype=torch.float32))
                self.labels.append(torch.tensor(label, dtype=torch.float32))

        self.data = torch.stack(self.data)
        self.labels = torch.stack(self.labels)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.labels[idx]

        # Data augmentation
        if self.augment and random.random() < 0.8:
            return add_gaussian_noise_standardized(data, std=self.noise), labels
        else:
            return data, labels


class FeatDataset4Cls(Dataset):
    """
    Hand-crafted feature samples for the FE-GNN / FE-MLP baselines (no augmentation).

    Args:
        dataset_dir: folder with *.pkl samples produced by tools/feat_write.py
        channel: modalities to keep, [0] = acceleration, [1] = strain
    """

    def __init__(self, dataset_dir, channel=[0], augment=False, LOGGER=None):
        self.dataset_dir = dataset_dir
        self.channel = channel
        self.augment = augment  # Not used

        self.feat = []
        self.labels = []
        self.files = glob(f"{self.dataset_dir}/*.pkl")

        if len(self.files) == 0:
            raise ValueError(f"No data is found in {self.dataset_dir}!")

        # Preload everything into memory
        for item_path in self.files:
            with open(item_path, "rb") as f:
                item_dict = pickle.load(f)
                feat = item_dict["feat"][self.channel]  # (1, feat_num, node_num)
                label = item_dict["label"]

                self.feat.append(torch.tensor(feat.copy(), dtype=torch.float32))
                self.labels.append(torch.tensor(label, dtype=torch.float32))

        self.feat = torch.stack(self.feat)
        self.labels = torch.stack(self.labels)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.feat[idx], self.labels[idx]
