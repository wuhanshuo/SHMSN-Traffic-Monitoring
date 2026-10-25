# SPDX-License-Identifier: AGPL-3.0-only
"""
Pre-compute the hand-crafted features of every training window for FEGNN / FEMLP.

    python -m tools.feat_write --data data/train --out data/train_feat

Each output pickle holds {"feat": float32 (2, 8, 8) = (modality, feature, node), "label": [...]}.
"""

import argparse
import pickle
from glob import glob
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from utils.feateng import feature_engineer

parser = argparse.ArgumentParser(description="Compute hand-crafted features of the training windows")
parser.add_argument("--data", type=str, default="data/train", help="Folder of training windows (*.pkl)")
parser.add_argument("--out", type=str, default="data/train_feat", help="Output folder for the feature samples")
args = parser.parse_args()

flist = glob(f"{args.data}/*.pkl")
print(f"There are {len(flist)} singals.")

output_folder = Path(args.out)
output_folder.mkdir(parents=True, exist_ok=True)

for fname in tqdm(flist):

    with open(fname, 'rb') as f:
        sample = pickle.load(f)
        signal = sample["signal"]
        label = sample["label"]

    # Split to acc and strain
    acc = torch.tensor(signal[0]).unsqueeze(0).unsqueeze(0)  # (1, 1, 500, 8)
    strain = torch.tensor(signal[1]).unsqueeze(0).unsqueeze(0)  # (1, 1, 500, 8)

    acc_feat = feature_engineer(acc)  # (1, 1, num_feat, 8)
    strain_feat = feature_engineer(strain)

    # to (2, num_feat, 8) numpy
    final_feat = torch.cat([acc_feat.squeeze(1), strain_feat.squeeze(1)], dim=0)
    final_feat_numpy = final_feat.cpu().numpy().astype(np.float32)

    output_file_path = output_folder / Path(fname).name
    with output_file_path.open("wb") as f:
        pickle.dump({"feat": final_feat_numpy, "label": label}, f)
