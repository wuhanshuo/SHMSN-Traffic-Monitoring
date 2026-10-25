# SPDX-License-Identifier: AGPL-3.0-only
"""
Train and test the CNN baseline (same encoder as the proposed model, no graph).

    python -m train.cnn --channel acc
    python -m train.cnn --channel strain

Results are written to runs/CNN_Reg_Train/train<N>/.
"""

import argparse
from datetime import datetime
from glob import glob

import torch

from utils.model import CNN
from utils.dataloader import YNLRegDataset4Cls
from utils.misc import MyLogger, seed_everything, create_dir
from utils.engine import run_multi_precision_test

parser = argparse.ArgumentParser(description="Train the CNN baseline on one modality")
parser.add_argument("--channel", type=str, help="Input channel: acc or strain", default="acc")
parser.add_argument("--noise", type=float, help="Std of the Gaussian noise augmentation", default=0.04)
parser.add_argument("--seed", type=int, help="Random seed", default=0)
parser.add_argument("--data", type=str, help="Folder of training windows (*.pkl)", default="data/train")
parser.add_argument("--testset", type=str, help="Folder of hourly test recordings (*.pkl)", default="data/test")
args = parser.parse_args()

# Initialize settings
result_path = create_dir("runs/CNN_Reg_Train", prefix="train")
seed_everything(args.seed)
LOGGER = MyLogger(result_path / "training.log")
LOGGER.log_device_info()
LOGGER.info(f"Experiment settings are {args}")

start_time = datetime.now()
LOGGER.info(f"The task started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

# Initialize the model
model = CNN()
model.log_arch(result_path / "architecture.log")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
LOGGER.info(f"The model is currently on: {next(model.parameters()).device}")
LOGGER.info(f"Model total parameters: {sum(p.numel() for p in model.parameters())}\n")

channel_dict = {"acc": [0], "strain": [1]}
channel = channel_dict[args.channel]

LOGGER.info(f"Dataset init started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
dataset = YNLRegDataset4Cls(args.data, channel=channel, noise=args.noise)
LOGGER.info(f"Dataset init ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

model.do_train(batch_size=256,
               dataset=dataset,
               epochs=500,
               lr=0.005,
               save_dir=result_path,
               device=device,
               patience=100,
               augment=True,
               LOGGER=LOGGER,
               )

end_time = datetime.now()
LOGGER.info(f"The task finished at {end_time.strftime(r'%Y-%m-%d %H:%M:%S')}")
LOGGER.info(f"Total execution time: {end_time-start_time}")

#########################
######## Test Signals
#########################
LOGGER = MyLogger(result_path / "test.log")
LOGGER.info(f"#############################")
LOGGER.info(f"Now let's test!")
model.load_state_dict(torch.load(result_path / "best.pth"))

test_list = sorted(glob(f"{args.testset}/*.pkl"))

# Run inference in FP32 (baseline), FP16 (autocast) and INT8 (dynamic-quantized Linear
# layers) and evaluate each precision independently (results/plots under
# result_path/<precision>/Test, aggregate metrics via eval_4cls/eval_2cls per precision).
run_multi_precision_test(model, test_list, channel, result_path, LOGGER, precisions=("fp32", "fp16", "int8"))

LOGGER.info(f"All results are successfully saved at: {result_path}")
