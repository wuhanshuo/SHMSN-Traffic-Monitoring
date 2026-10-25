# SPDX-License-Identifier: AGPL-3.0-only
"""
Evaluate already-trained CNN + GNN checkpoints in FP32 / FP16 / INT8 without retraining.

    python -m evaluation.precision --acc_dir runs/singleacc --strain_dir runs/singlestr --dual_dir runs/dual

Each *_dir is a run folder produced by single.py (acc / strain) or dual.py and must contain
best.pth. The folders are only read; all outputs go to runs/Precision_Eval/eval<N>/.
"""

import argparse
from pathlib import Path
from glob import glob

import torch

from utils.model import ModelRegGNN, DualRegGNN
from utils.misc import MyLogger, create_dir
from utils.engine import run_multi_precision_test

parser = argparse.ArgumentParser(description="Multi-precision evaluation of trained checkpoints")
parser.add_argument("--acc_dir", type=str, help="Run folder of ModelRegGNN trained on accelerations", default="runs/singleacc")
parser.add_argument("--strain_dir", type=str, help="Run folder of ModelRegGNN trained on strains", default="runs/singlestr")
parser.add_argument("--dual_dir", type=str, help="Run folder of DualRegGNN", default="runs/dual")
parser.add_argument("--testset", type=str, help="Folder of hourly test recordings (*.pkl)", default="data/test")
parser.add_argument("--precisions", type=str, nargs="+", help="Any of fp32 fp16 int8", default=["fp32", "fp16", "int8"])
args = parser.parse_args()

# Initialize settings
result_path = create_dir("runs/Precision_Eval", prefix="eval")
LOGGER = MyLogger(result_path / "test.log")
LOGGER.log_device_info()
LOGGER.info(f"Experiment settings are {args}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_list = sorted(glob(f"{args.testset}/*.pkl"))
if not test_list:
    LOGGER.error(f"No test signal (*.pkl) found under {args.testset}!")
LOGGER.info(f"Found {len(test_list)} test signals under {args.testset}\n")

run_configs = [
    ("acc", ModelRegGNN(graph="ladder"), [0], Path(args.acc_dir)),
    ("strain", ModelRegGNN(graph="ladder"), [1], Path(args.strain_dir)),
    ("dual", DualRegGNN(graph="ladder"), [0, 1], Path(args.dual_dir)),
]

for name, model, channel, run_dir in run_configs:
    ckpt = run_dir / "best.pth"
    if not ckpt.exists():
        LOGGER.error(f"[{name}] Checkpoint not found at {ckpt}, skipping this model.")
        continue

    LOGGER.info(f"#############################")
    LOGGER.info(f"=== [{name}] Loading checkpoint from {ckpt} ===")
    model.to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    LOGGER.info(f"[{name}] Model total parameters: {sum(p.numel() for p in model.parameters())}\n")

    eval_dir = result_path / name
    run_multi_precision_test(model, test_list, channel, eval_dir, LOGGER, precisions=tuple(args.precisions))

LOGGER.info(f"All precision evaluation results are saved at: {result_path}")
