# SPDX-License-Identifier: AGPL-3.0-only
"""
Hand-crafted features for the FE-GNN / FE-MLP baselines.

Eight statistics are computed along the time axis of every sensor node:
min, max, mean, std, kurtosis, RMS, absolute sum and energy.
Input (B, C, T, N) -> output (B, C, 8, N).
"""

import torch
from scipy.stats import kurtosis
import numpy as np


# Compute the minimum value for each sensor (node)
def compute_min(data):
    return torch.min(data, dim=2)[0].unsqueeze(-1)

# Compute the maximum value for each sensor (node)
def compute_max(data):
    return torch.max(data, dim=2)[0].unsqueeze(-1)

# Compute the mean value for each sensor (node)
def compute_mean(data):
    return torch.mean(data, dim=2).unsqueeze(-1)

# Compute the standard deviation for each sensor (node)
def compute_std(data):
    return torch.std(data, dim=2, unbiased=False).unsqueeze(-1)


# Compute the kurtosis for each sensor (node)
def compute_kurtosis(data):
    # Convert torch tensor to numpy array for kurtosis calculation
    data_numpy = data.cpu().numpy()
    kurtosis_values = np.apply_along_axis(kurtosis, axis=2, arr=data_numpy)
    return torch.tensor(kurtosis_values, dtype=torch.float32, device=data.device).unsqueeze(-1)

# Compute the root mean square (RMS) for each sensor (node)
def compute_rms(data):
    return torch.sqrt(torch.mean(data**2, dim=2)).unsqueeze(-1)

# Compute the absolute sum for each sensor (node)
def compute_abs_sum(data):
    return torch.sum(torch.abs(data), dim=2).unsqueeze(-1)

# Compute the energy for each sensor (node)
def compute_energy(data):
    return torch.sum(data**2, dim=2).unsqueeze(-1)

def feature_engineer(data):
    """Stack the eight statistics along a new feature axis: (B, C, T, N) -> (B, C, 8, N)."""
    features = torch.cat([
        compute_min(data), 
        compute_max(data), 
        compute_mean(data),
        compute_std(data),
        compute_kurtosis(data), 
        compute_rms(data), 
        compute_abs_sum(data), 
        compute_energy(data)
    ], dim=-1)
    return features.float()  # kurtosis comes back from numpy; keep everything float32


if __name__ == "__main__":
    data = torch.rand(32, 1, 500, 8)
    print(feature_engineer(data).shape)  # (32, 1, 8, 8)
