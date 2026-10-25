# SPDX-License-Identifier: AGPL-3.0-only
"""
Inference-time benchmark of the peak-detection baseline and the trained models.

    python -m evaluation.speed_test              # GPU if available
    python -m evaluation.speed_test --device cpu

Every model is run on the full test set once per input channel (acceleration, strain).
Only the timing is of interest here: the same checkpoint is used for both channels, so the
predictions themselves are not meaningful. Logs are written to runs/Speed_Test/test.log.
"""

import argparse
import pickle
import time
from datetime import datetime
from glob import glob
from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks

from utils.model import ModelRegGNN, CNN, FEGNN, FEMLP
from utils.misc import MyLogger

parser = argparse.ArgumentParser(description="Inference-time benchmark")
parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--testset", type=str, help="Folder of hourly test recordings (*.pkl)", default="data/test")
parser.add_argument("--gnn", type=str, help="ModelRegGNN checkpoint", default="runs/singleacc/best.pth")
parser.add_argument("--cnn", type=str, help="CNN checkpoint", default="runs/cnnacc/best.pth")
parser.add_argument("--fegnn", type=str, help="FEGNN checkpoint", default="runs/fegnnacc/best.pth")
parser.add_argument("--femlp", type=str, help="FEMLP checkpoint", default="runs/femlpacc/best.pth")
args = parser.parse_args()

result_path = Path("runs/Speed_Test")
result_path.mkdir(parents=True, exist_ok=True)
LOGGER = MyLogger(result_path / "test.log")
LOGGER.log_device_info()
LOGGER.info(f"Experiment settings are {args}")
LOGGER.info(f"The task started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

device = torch.device(args.device)
test_list = sorted(glob(f"{args.testset}/*.pkl"))


def predict_strain_rule(signal_path, LOGGER):
    """
    Rule-based baseline: count strain peaks of one sensor per lane (scipy find_peaks) and bin
    them into WINDOW-long windows. Returns an array (num_windows, 4) like predict_long().
    The height ranges are the calibrated thresholds of Table A.1 in the paper.
    """
    with open(signal_path, 'rb') as f:
        test_signal = pickle.load(f)["signal"]

    strain_left = test_signal[1, :, 1]   # sensor above the left (overtaking) lane
    strain_right = test_signal[1, :, 6]  # sensor above the right (slow) lane

    total_len = test_signal.shape[1]
    if total_len % 500 != 0:
        LOGGER.warning(f"Signal length {total_len} is not a multiple of 500.")
    piece_num = total_len // 500

    peaks_car_left, _ = find_peaks(strain_left, height=(0.04, 0.4), distance=10)
    peaks_car_right, _ = find_peaks(strain_right, height=(0.035, 0.1), distance=10)
    peaks_truck_left, _ = find_peaks(strain_left, height=(0.4, 1.0), distance=10)
    peaks_truck_right, _ = find_peaks(strain_right, height=(0.1, 1.0), distance=10)

    # Bin the peak indices into the 500-sample windows
    output_raw = np.zeros((piece_num, 4))
    for col_idx, peaks in enumerate([peaks_car_left, peaks_car_right, peaks_truck_left, peaks_truck_right]):
        for p in peaks:
            bin_idx = p // 500
            if bin_idx < piece_num:
                output_raw[bin_idx, col_idx] += 1
    return output_raw


def log_timing(start_time):
    LOGGER.info(f"Total samples in the testset: {len(test_list)}")
    LOGGER.info(f"Total inference time on the testset is: {time.time() - start_time} seconds")


# Peak detection
LOGGER.info("Testing peak detection")
start_time = time.time()
for fname in test_list:
    predict_strain_rule(fname, LOGGER=LOGGER)
log_timing(start_time)

# Learned models
benchmarks = [
    ("CNN+GNN", ModelRegGNN(graph="ladder"), args.gnn),
    ("CNN", CNN(), args.cnn),
    ("FEGNN", FEGNN(graph="ladder"), args.fegnn),
    ("FEMLP", FEMLP(), args.femlp),
]
channel_dict = {"acc": [0], "strain": [1]}

for name, model, ckpt in benchmarks:
    LOGGER.info(f"Testing {name}")
    model.device = device  # predict_long() runs on model.device
    model.to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    for k, v in channel_dict.items():
        LOGGER.info(f"Using channel: {k}")
        start_time = time.time()
        for test_signal in test_list:
            model.predict_long(test_signal, save_dir=result_path / "Test", channel=v, LOGGER=LOGGER)
        log_timing(start_time)
