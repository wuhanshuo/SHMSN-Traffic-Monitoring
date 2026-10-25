# SPDX-License-Identifier: AGPL-3.0-only
"""
Build the training windows from the raw recordings and the CV-derived vehicle labels.

    python -m tools.generator --raw <folder of *.pickle recordings> --labels data/label_axle_corr.csv --out data/train

Recordings from day 25-29 (08:00-17:59) are cut into 5 s windows (500 samples at 100 Hz)
with a 2.5 s stride. Each window is labelled with fractional vehicle counts
[LeftCar, RightCar, LeftTruck, RightTruck]: a vehicle that is only partly inside the window
contributes the fraction of its passage that falls into it. The label CSV is created in
notebooks/DatasetGenerator.ipynb (columns: start, end, type, x, speed, axle).
"""

import argparse
import pickle
from glob import glob
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from utils.composer import read_pickle

parser = argparse.ArgumentParser(description="Generate labelled training windows")
parser.add_argument("--raw", type=str, required=True, help="Folder of raw *.pickle recordings")
parser.add_argument("--labels", type=str, default="data/label_axle_corr.csv", help="Vehicle label CSV")
parser.add_argument("--out", type=str, default="data/train", help="Output folder for the *.pkl windows")
args = parser.parse_args()

signal_list = sorted(glob(args.raw + "/*.pickle"))

# Recording files are named <site>_<start yyyymmddHHMMSS>_<end yyyymmddHHMMSS>.pickle
filtered_list = []
for i in signal_list:
    ft = pd.to_datetime(i.split("_")[-2], format='%Y%m%d%H%M%S')
    if ft.day >= 25 and ft.day <= 29 and ft.hour >= 8 and ft.hour <= 17:
        filtered_list.append(i)
print(f"Filtered list has length = {len(filtered_list)}")

window_size = 500  # samples (5 s)
step = 250  # samples (2.5 s)
print("Overlapping = ", 1 - step / window_size)

df_label = pd.read_csv(args.labels, parse_dates=["start", "end"])

output_folder = Path(args.out)
output_folder.mkdir(parents=True, exist_ok=True)

for f in tqdm(filtered_list):
    tqdm.write(f"Processing {f.split('_')[-2]}")
    dual_np, start, end = read_pickle(f, norm=True, signal_filter=True)
    # Drop 2.5 s at both ends to discard filter edge effects
    dual_np = dual_np[:, 250:-250, :]
    start = start + pd.Timedelta(seconds=2.5)
    end = end - pd.Timedelta(seconds=2.5)

    piece_num = (dual_np.shape[1] - (window_size - step)) // step

    for i in range(piece_num):

        sample = dual_np[:, i * step:i * step + window_size, :]
        sample_start = start + pd.Timedelta(seconds=step / 100) * i
        sample_end = sample_start + pd.Timedelta(seconds=window_size / 100)

        # Vehicles whose passage overlaps the window
        within_vehicles = df_label[(df_label["start"] >= sample_start) & (df_label["start"] <= sample_end) |
                                   (df_label["end"] >= sample_start) & (df_label["end"] <= sample_end)]
        reg_label = [0, 0, 0, 0]  # (CL, CR, TL, TR)
        for v in within_vehicles.itertuples():
            lane = 0 if v.x < 7 else 1  # lateral position x [m]: < 7 left lane, otherwise right lane
            obj_type = v.type
            duration = v.end - v.start

            if v.start >= sample_start and v.end <= sample_end:
                vc = 1
            elif v.start < sample_start and v.end < sample_end:
                vc = (v.end - sample_start) / duration
            elif v.start > sample_start and v.end > sample_end:
                vc = (sample_end - v.start) / duration
            else:
                print(f"Impossible exception: vehicle from {v.start} to {v.end}. This sample is from {sample_start}.")

            if obj_type == 1:    # car
                if lane == 0:  # left lane
                    reg_label[0] += vc
                else:           # right lane
                    reg_label[1] += vc
            elif obj_type == 2:  # truck
                if lane == 0:
                    reg_label[2] += vc
                else:
                    reg_label[3] += vc
            else:
                raise ValueError(f"Unknown vehicle type {obj_type}")

        sample_name = sample_start.strftime("%Y%m%d%H%M%S")

        with open(output_folder / f"{sample_name}.pkl", "wb") as f:
            pickle.dump({"signal": sample, "label": reg_label}, f)
