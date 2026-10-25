# SPDX-License-Identifier: AGPL-3.0-only
"""Run-folder creation, logging and reproducibility helpers."""

from pathlib import Path
import logging
import torch


def create_dir(parent_dir, prefix="run"):
    """
    Create a folder with given prefix and incrementing number as suffixes.
    Args:
        parent_dir: The parent directory to create the folder.
        prefix: The prefix of the folder name.
    """
    run_num = 1
    while run_num <= 100:
        dir = Path(f"{parent_dir}/{prefix}{run_num}")
        if not dir.exists():
            dir.mkdir(parents=True, exist_ok=True)
            break
        run_num += 1
    return dir


class MyLogger(logging.Logger):
    """Logger writing to both `log_file` and the console (INFO messages without level prefix)."""

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            if record.levelname == "INFO":
                self._fmt = '%(message)s'
            else:
                self._fmt = '%(levelname)s: %(message)s'
            return super().format(record)

    def __init__(self, log_file="logs/app.log"):
        super().__init__(__name__, logging.INFO)

        if not self.hasHandlers():  # Avoid duplicate handler
            log_file_path = Path(log_file)
            log_file_path.parent.mkdir(parents=True, exist_ok=True)

            fh = logging.FileHandler(log_file_path)
            fh.setLevel(logging.INFO)

            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)

            formatter = self.CustomFormatter()
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)

            self.addHandler(fh)
            self.addHandler(ch)

    def log_device_info(self):
        """Log GPU information"""
        self.info(f"Torch version: {torch.__version__}")
        if torch.cuda.is_available():
            self.info(f"Cuda version: {torch.version.cuda}")
            self.info(f"Cuda available: {torch.cuda.is_available()}")
            self.info(f"Cuda device count: {torch.cuda.device_count()}")
            self.info(f"Cuda current device: {torch.cuda.current_device()}")
            self.info(f"Cuda current device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        else:
            self.info("No GPU available.")


def monitor_gpu(LOGGER):
    """Log current / peak CUDA memory usage and reset the peak counter."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**2
        reserved_memory = torch.cuda.memory_reserved(device) / 1024**2
        allocated_memory = torch.cuda.memory_allocated(device) / 1024**2
        free_memory = reserved_memory - allocated_memory

        LOGGER.info(f"Total memory: {total_memory:.2f} MB")
        LOGGER.info(f"Reserved memory: {reserved_memory:.2f} MB")
        LOGGER.info(f"Allocated memory: {allocated_memory:.2f} MB")
        LOGGER.info(f"Free memory: {free_memory:.2f} MB")
        LOGGER.info(f"Peak memory: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")

        torch.cuda.reset_peak_memory_stats()


def seed_everything(seed):
    """
    Fix the random seeds of random / numpy / torch and make cuDNN deterministic.
    Args:
        seed: int
    """
    import random, os
    import numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # Fix the randomness of Python hash (may not always work)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # Use deterministic convolution algorithms
    torch.backends.cudnn.benchmark = False  # Disable auto-tuning, which may pick non-deterministic kernels
