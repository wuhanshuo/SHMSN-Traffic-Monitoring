# SPDX-License-Identifier: AGPL-3.0-only
"""
Models for lane-specific traffic-volume regression from SHM sensor networks.

All models map a 5 s window of the sensor network (8 nodes x 500 time steps at 100 Hz)
to four (fractional) vehicle counts:  [LeftCar, RightCar, LeftTruck, RightTruck].

    ModelRegGNN  CNN encoder (YOLO11 C3k2 backbone) + 2x GATv2 + linear   (one modality)
    DualRegGNN   one CNN encoder + GATv2 branch per modality (acc, strain), fused before readout
    CNN          CNN encoder + linear (no graph)                            (baseline)
    FEGNN        hand-crafted features + 2x GATv2 + linear                  (baseline)
    FEMLP        hand-crafted features + 3-layer MLP                        (baseline)

Tensor layout: (batch, channel, time, node), i.e. (B, 1, 500, 8) for one modality.
"""

from pathlib import Path
import numpy as np
import copy
import time
import pickle
import contextlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

import dgl
from dgl.nn import GATv2Conv, AvgPooling

from .dataloader import collate_fn
from .engine import create_graph, MultiTaskLoss, save_csv, plot_loss, plot_r2, plot_error, plot_lr
from .metrics import coeff_determination, l1_loss_detach, mse_loss_detach, l1_loss_detach_np
from .feateng import feature_engineer
from .modules import Conv, C3k2

CLS_DICT = {0: "LeftCar", 1: "RightCar", 2: "LeftTruck", 3: "RightTruck"}
WINDOW = 500  # samples per inference window (5 s at 100 Hz)


def yolo_encoder(in_channel, pool=(16, 1)):
    """
    YOLO11-style convolutional encoder operating along the time axis only.

    (B, in_channel, 500, N) -> five stride-(2, 1) stages -> (B, 1280, 16, N) -> max-pool over time.
    With pool=(16, 1) the output is one 1280-d embedding per sensor node, (B, 1280, 1, N);
    with pool=(16, N) the nodes are pooled as well (used by the plain CNN baseline).
    """
    return nn.Sequential(
        Conv(in_channel, 16, k=(3, 1), s=(2, 1), p=(1, 0)),
        Conv(16, 32, k=(3, 1), s=(2, 1), p=(1, 0)),
        C3k2(32, 32, n=1, shortcut=True),
        Conv(32, 64, k=(3, 1), s=(2, 1), p=(1, 0)),
        C3k2(64, 64, n=2, shortcut=True),
        Conv(64, 128, k=(3, 1), s=(2, 1), p=(1, 0)),
        C3k2(128, 128, n=2, shortcut=True),
        Conv(128, 256, k=(3, 1), s=(2, 1), p=(1, 0)),
        C3k2(256, 256, n=1, shortcut=True),
        Conv(256, 1280, 1, 1, None, 1),  # rich feature, (B, 1280, 16, N)
        nn.MaxPool2d(kernel_size=pool),
    )


class ModelRegGNN(nn.Module):
    """
    CNN + GNN regressor for one sensing modality (the proposed model).

    Each sensor node is encoded independently by the CNN, the node embeddings are exchanged
    over the sensor graph by two GATv2 layers (8 heads each), averaged into a graph embedding
    and mapped to the four traffic counts. Training/inference utilities shared by all models
    (do_train, predict_long, precision handling) are defined here.
    """

    def __init__(self, graph="ladder", num_class=4, in_channel=1):
        super().__init__()
        self.num_class = num_class
        self.graph = graph
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Feature extractor: (B, 1, 500, 8) -> (B, 1280, 1, 8)
        self.encoder = yolo_encoder(in_channel)

        # Message passing over the sensor graph
        self.gat1 = GATv2Conv(in_feats=1280, out_feats=128, num_heads=8, activation=nn.SiLU())
        self.gat4 = GATv2Conv(in_feats=1024, out_feats=128, num_heads=8, activation=nn.SiLU())

        # Readout
        self.graph_readout = AvgPooling()
        self.linear = nn.Linear(1024, num_class)

    def forward(self, x, g):
        x8 = self.encoder(x)

        sensor_feats = x8.permute(0, 3, 1, 2)  # (B, 8, 1280, 1)
        batch_size, num_nodes, feat_num, feat_size = sensor_feats.shape
        sensor_feats = sensor_feats.reshape(batch_size * num_nodes, feat_num, feat_size)  # (B*8, 1280, 1)

        g_in = sensor_feats.flatten(1)  # (B*8, 1280)
        g_batched = dgl.batch([g] * batch_size)

        h = self.gat1(g_batched, g_in).flatten(1)  # (B*8, 8 heads * 128)
        g_out = self.gat4(g_batched, h).flatten(1)  # (B*8, 1024)

        gf = self.graph_readout(g_batched, g_out)  # (B, 1024)
        return self.linear(gf)

    def do_train(self, batch_size, dataset, epochs, lr, save_dir, device=None, patience=100, augment=False, LOGGER=None):
        """
        Train the model with a random 80/20 train/validation split of `dataset`.

        Loss: per-class MSE (summed over the batch) weighted by learnable homoscedastic
        uncertainty (Kendall et al., 2018). Optimizer: AdamW with gradient clipping (5.0).
        Schedule: cosine annealing with warm restarts (30 warm-up epochs, 100-epoch cycles).
        `last.pth` is written every epoch and `best.pth` whenever the validation loss improves;
        training stops early after `patience` epochs without improvement.
        """
        if LOGGER is None:
            import logging
            LOGGER = logging.getLogger(__name__)

        save_dir = Path(save_dir)
        LOGGER.info(f"The training result is saved at: {save_dir}")
        LOGGER.info(f"Graph type: {self.graph}")

        if device is None:
            device = self.device
        self.to(device)

        # reduction="none" keeps the loss as (batch, num_class) so each class is weighted separately
        criterion = nn.MSELoss(reduction="none")
        weigh_loss = MultiTaskLoss(num_tasks=self.num_class).to(device)
        LOGGER.info(f"Criteria: {type(criterion).__name__}")

        optimizer = optim.AdamW(list(self.parameters()) + list(weigh_loss.parameters()), lr=lr)
        LOGGER.info(f"Optimizer: {type(optimizer).__name__}")
        LOGGER.info(f"Total parameters in optimizer: {sum(p.numel() for pg in optimizer.param_groups for p in pg['params'])}")

        scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer, first_cycle_steps=100, cycle_mult=1.0, max_lr=lr,
                                                  min_lr=0.1e-6, warmup_steps=30, gamma=0.5)
        LOGGER.info(f"Scheduler: {type(scheduler).__name__}")
        LOGGER.info(f"Initial learning rate: {lr}\n")

        # Random 80/20 split
        val_size = len(dataset) // 5
        train_size = len(dataset) - val_size
        if val_size == 0:
            LOGGER.error(f"The dataset {dataset} is not found!")
        train_set, val_set = random_split(dataset, [train_size, val_size])
        LOGGER.info(f"The model is trained on the dataset: {str(dataset.dataset_dir)}\n")
        LOGGER.info(f"The dataset is successfully loaded, channel {dataset.channel} is applied, and {dataset[0][0].size(-1)} sensors are applied.")

        # NOTE: both subsets share the underlying dataset object, so the augmentation flag
        # set here is seen by the validation subset as well.
        train_set.dataset.augment = augment

        if train_set.dataset.augment:
            LOGGER.info(f"Training set agumentation is applied.")
        else:
            LOGGER.info(f"Training set agumentation is NOT applied.")

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2, persistent_workers=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2, persistent_workers=True)

        LOGGER.info(f"Length of training set is {len(train_set)}")
        LOGGER.info(f"Length of validation set is {len(val_set)}\n")
        if len(train_set) != train_size:
            raise ValueError(f"The size of train_set should be {train_size}, got {len(train_set)} instead.")
        elif len(val_set) != val_size:
            raise ValueError(f"The size of val_set should be {val_size}, got {len(val_set)} instead.")

        # Training history
        train_loss_list = []
        val_loss_list = []
        metrics_dict = {
            "MAE": [],
            "MSE": []
        }
        train_r2_list = []
        val_r2_list = []
        lr_list = []
        best_index = 0

        for epoch in range(epochs):

            epoch_start = time.time()
            LOGGER.info(f"Epoch [{epoch}/{epochs}]")

            # ---- Training ----
            self.train()
            train_loss = 0.0
            train_r2_epoch = []

            for (signals, labels) in train_loader:
                signals_batch = torch.stack(signals, dim=0).to(device)
                labels_batch = torch.stack(labels, dim=0).to(device)

                optimizer.zero_grad()

                if self.graph is not None:
                    g = create_graph(signals_batch.size(-1), self.graph).to(device)
                    outputs = self.forward(signals_batch, g)
                else:
                    outputs = self.forward(signals_batch)

                loss = criterion(outputs, labels_batch)
                loss = weigh_loss(loss.sum(dim=0))  # sum over the batch, weight per class

                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
                optimizer.step()
                train_loss += loss.item()

                train_r2_epoch.append(coeff_determination(outputs.detach(), labels_batch))

            train_loss /= train_size
            train_loss_list.append(train_loss)
            # Average R2 over batches, ignoring NaN (classes without variance in a batch)
            train_r2_list.append(torch.nanmean(torch.stack(train_r2_epoch), dim=0).cpu().numpy())

            # ---- Validation ----
            self.eval()
            val_loss = 0.0
            val_MAE = np.zeros((1, self.num_class))
            val_MSE = np.zeros((1, self.num_class))
            val_r2_epoch = []
            with torch.no_grad():
                for (signals, labels) in val_loader:
                    signals_batch = torch.stack(signals, dim=0).to(device)
                    labels_batch = torch.stack(labels, dim=0).to(device)
                    if self.graph is not None:
                        g = create_graph(signals_batch.size(-1), self.graph).to(device)
                        outputs = self.forward(signals_batch, g)
                    else:
                        outputs = self.forward(signals_batch)

                    loss = criterion(outputs, labels_batch)
                    loss = weigh_loss(loss.sum(dim=0))
                    val_loss += loss.item()
                    val_MAE += l1_loss_detach(outputs, labels_batch).cpu().numpy()
                    val_MSE += mse_loss_detach(outputs, labels_batch).cpu().numpy()
                    val_r2_epoch.append(coeff_determination(outputs, labels_batch))

            val_loss /= val_size
            val_loss_list.append(val_loss)
            val_MAE /= val_size
            val_MSE /= val_size
            metrics_dict["MAE"].append(val_MAE[0])
            metrics_dict["MSE"].append(val_MSE[0])
            val_r2_list.append(torch.nanmean(torch.stack(val_r2_epoch), dim=0).cpu().numpy())

            # ---- Learning-rate schedule ----
            if isinstance(scheduler, CosineAnnealingWarmupRestarts):
                lr_list.append(scheduler.get_lr()[0])
            else:
                lr_list.append(scheduler.get_last_lr()[0])

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

            # ---- Checkpointing ----
            torch.save(self.state_dict(), save_dir / "last.pth")
            if val_loss <= min(val_loss_list):
                best_index = epoch
                torch.save(self.state_dict(), save_dir / "best.pth")

            sec = time.time() - epoch_start

            LOGGER.info(f"lr: {lr_list[-1]}, Loss Type: {type(criterion).__name__}")
            LOGGER.info(f"Train Loss: {train_loss}, Val Loss: {val_loss}")
            LOGGER.info(f"Train R2: {train_r2_list[-1]}, Val R2: {val_r2_list[-1]}")
            LOGGER.info(f"Detached Loss on Validation Set: MAE {val_MAE}, MSE {val_MSE}")
            LOGGER.info(f"Execution Time: {sec} sec\n")

            if epoch >= best_index + patience:
                LOGGER.info(f"Early stop at eopch {epoch}. The best model is found at epoch {best_index}")
                break

        LOGGER.info(f"The best model is found at epoch {best_index}, validation loss is {val_loss_list[best_index]}")
        LOGGER.info(f"The best model has MAE: {metrics_dict['MAE'][best_index]}, MSE: {metrics_dict['MSE'][best_index]}\n")

        save_csv(lr_list, train_loss_list, val_loss_list, metrics_dict, train_r2_list, val_r2_list, save_dir=save_dir)

        plot_loss(train_loss_list, val_loss_list, criterion_name=type(criterion).__name__, save_dir=save_dir)
        plot_r2(train_r2_list, val_r2_list, save_dir=save_dir)
        plot_error(metrics_dict, save_dir=save_dir)
        plot_lr(lr_list, save_dir=save_dir)

        LOGGER.info(f"Training results are successfully saved at {save_dir}.")

    def load_state_dict(self, *args, **kwargs):
        """Loading fresh weights invalidates any cached INT8 copy built from stale weights."""
        object.__setattr__(self, "_int8_model", None)
        return super().load_state_dict(*args, **kwargs)

    def _get_precision_context(self, precision="fp32"):
        """
        Prepare an inference-ready model, its device, and an autocast context manager
        for the requested precision. Returns (model, device, autocast_ctx).

        - "fp32": no-op, runs the model as-is on its native device.
        - "fp16": mixed-precision autocast. On CUDA this runs eligible ops (Linear/Conv)
          in true float16 (BatchNorm is kept in fp32 by autocast for stability). On CPU,
          true fp16 kernels are largely unimplemented, so bfloat16 autocast is used instead
          as the closest practical stand-in for reduced-precision CPU inference.
        - "int8": post-training dynamic quantization (weight-only, nn.Linear layers only,
          via PyTorch's CPU int8 kernels). PyTorch's dynamic quantization does not support
          Conv2d, so the convolutional encoder stays in fp32; only Linear layers (the
          decoder head, and any Linear layers inside the GNN blocks) are quantized. This
          requires the CPU backend, so the returned device is always "cpu" for this branch.
          The quantized copy is cached on the instance and rebuilt automatically after
          load_state_dict() is called.
        """
        precision = precision.lower()
        if precision == "fp32":
            return self, self.device, contextlib.nullcontext()
        elif precision == "fp16":
            if self.device.type == "cuda":
                return self, self.device, torch.autocast(device_type="cuda", dtype=torch.float16)
            else:
                return self, self.device, torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        elif precision == "int8":
            if getattr(self, "_int8_model", None) is None:
                cpu_model = copy.deepcopy(self).to("cpu").eval()
                quant_model = torch.ao.quantization.quantize_dynamic(
                    cpu_model, {nn.Linear}, dtype=torch.qint8
                )
                object.__setattr__(self, "_int8_model", quant_model)
            return self._int8_model, torch.device("cpu"), contextlib.nullcontext()
        else:
            raise ValueError(f"Unknown precision '{precision}', expected one of: fp32, fp16, int8")

    def _finalize_predictions(self, output_raw, test_signal, test_label, fname, save_dir, precision, LOGGER):
        """Shared MAE computation, plotting and artifact saving for predict_long()."""

        save_dir = Path(save_dir)
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)

        num_cls = test_label.shape[-1]

        # Guard against a signal length that is not an exact multiple of WINDOW: keep the
        # overlapping range so MAE/plots never crash on a shape mismatch.
        min_len = min(output_raw.shape[0], test_label.shape[0])
        output_raw = output_raw[:min_len]
        test_label = test_label[:min_len]

        # Counts cannot be negative; clipping is the only post-processing of the raw output
        output_norm = np.clip(output_raw.copy(), 0, 8)
        test_MAE = l1_loss_detach_np(output_norm, test_label) / max(min_len, 1)

        for i in range(num_cls):
            plt.close()
            plt.clf()
            plt.figure(figsize=(20, 8))
            plt.suptitle(f"[{precision}] {CLS_DICT[i]} MAE: {test_MAE[i]}")

            plt.subplot(4, 1, 1)
            squeeze = test_label[:, i].flatten()
            plt.plot(squeeze)
            plt.ylim(0, 6)
            plt.title(f"Ground Truth - {CLS_DICT[i]}: {(sum(squeeze)):.3f}")

            plt.subplot(4, 1, 2)
            plt.plot(output_raw[:, i])
            plt.ylim(-1, 6)
            out = output_raw[:, i].sum()
            plt.title(f"Model Raw Output - {CLS_DICT[i]}: {out:.3f}")

            plt.subplot(4, 1, 3)
            plt.plot(output_norm[:, i])
            plt.ylim(0, 6)
            out = output_norm[:, i].sum()
            plt.title(f"Model Normalized Output - {CLS_DICT[i]}: {out:.3f}")
            LOGGER.info(f" - [{precision}] Class {i}: GT = {(sum(squeeze)):.3f}, PREDICTION = {out:.3f}")

            plt.subplot(4, 1, 4)
            length = test_signal.shape[1]
            if i % 2 == 0:
                plt.plot(np.arange(length), test_signal[0, :, 0])
            else:
                plt.plot(np.arange(length), test_signal[0, :, -1])
            plt.title("Signal")

            plt.tight_layout()
            plt.savefig(save_dir / f"{fname}_Test_{CLS_DICT[i]}.png")
            plt.close()

        LOGGER.info(f" - [{precision}] MAE of the signal is: {test_MAE}.")
        np.save(save_dir / f"{fname}_raw.npy", output_raw)
        np.save(save_dir / f"{fname}_norm.npy", output_norm)
        np.save(save_dir / f"{fname}_gt.npy", test_label)

        return output_raw, test_MAE

    def _preprocess(self, signal_tensor):
        """Map a raw (1, C, WINDOW, N) window to the model input. Identity for end-to-end models."""
        return signal_tensor

    def predict_long(self, signal_path, save_dir, channel=[0], LOGGER=None, precision="fp32"):
        """
        Run inference on a long test recording.

        The pickle at `signal_path` holds {"signal": (2, T, 8), "label": (T // WINDOW, 4)}. The
        signal is cut into non-overlapping windows of WINDOW samples, each window is predicted
        independently, and plots plus *_raw/_norm/_gt.npy files are written to `save_dir`.
        Returns (raw predictions of shape (num_windows, 4), per-class MAE).
        """
        self.eval()

        if LOGGER is None:
            import logging
            LOGGER = logging.getLogger(__name__)

        fname = Path(signal_path).stem

        with open(signal_path, 'rb') as f:
            pk = pickle.load(f)
            test_signal = pk["signal"][channel]
            test_label = pk["label"]

        if test_signal.shape[1] % WINDOW != 0:
            LOGGER.warning(f"Warning! Test signal has shape {test_signal.shape}, not a multiple of {WINDOW}!")
        piece_num = test_signal.shape[1] // WINDOW

        infer_model, device, autocast_ctx = self._get_precision_context(precision)

        # Move the whole recording to the target device once instead of once per window
        full_signal = torch.tensor(test_signal, dtype=torch.float32).to(device)

        # The graph only depends on the (fixed) number of sensors, so build it once
        g_dev = None
        if self.graph is not None:
            g_dev = create_graph(test_signal.shape[-1], self.graph).to(device)

        output_list = []
        for s in range(piece_num):
            # Slice on the target device and add the batch dimension: (1, C, WINDOW, N)
            signal_tensor = full_signal[:, s * WINDOW: s * WINDOW + WINDOW, :].unsqueeze(0)

            # Kept outside the precision context: feature_engineer() always returns float32
            model_input = self._preprocess(signal_tensor)

            with torch.no_grad(), autocast_ctx:
                if self.graph is not None:
                    output = infer_model.forward(model_input, g_dev)[0]
                else:
                    output = infer_model.forward(model_input)[0]

            output_list.append(output.float().cpu().numpy())  # (num_cls,)

        output_raw = np.array(output_list)  # (num_windows, num_cls)

        return self._finalize_predictions(output_raw, test_signal, test_label, fname, save_dir, precision, LOGGER)

    def log_arch(self, target):
        """Write the module tree to `target`."""
        with open(target, "w") as fs:
            fs.write(str(self))


class DualRegGNN(ModelRegGNN):
    """
    CNN + GNN regressor fusing accelerometers and strain gauges.

    Each modality has its own CNN encoder and its own two GATv2 layers. The two node-level
    outputs (512-d each) are concatenated, averaged over the graph and mapped to the counts.
    Input: (B, 2, 500, 8) with channel 0 = acceleration, channel 1 = strain.
    """

    def __init__(self, graph="ladder", num_class=4, in_channel=(1, 1)):
        nn.Module.__init__(self)
        self.num_class = num_class
        self.graph = graph
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Feature extractors, one per modality: (B, 1, 500, 8) -> (B, 1280, 1, 8)
        self.encoder_acc = yolo_encoder(in_channel[0])
        self.encoder_strain = yolo_encoder(in_channel[1])

        # Message passing, one branch per modality
        self.gat1_a = GATv2Conv(in_feats=1280, out_feats=128, num_heads=8, activation=nn.SiLU())
        self.gat3_a = GATv2Conv(in_feats=1024, out_feats=64, num_heads=8, activation=nn.SiLU())

        self.gat1_s = GATv2Conv(in_feats=1280, out_feats=128, num_heads=8, activation=nn.SiLU())
        self.gat3_s = GATv2Conv(in_feats=1024, out_feats=64, num_heads=8, activation=nn.SiLU())

        # Readout
        self.graph_readout = AvgPooling()
        self.linear = nn.Linear(1024, num_class)

    def forward(self, x, g):
        x1, x2 = torch.split(x, [1, 1], dim=1)  # (B, 1, 500, 8) each

        x_acc = self.encoder_acc(x1).permute(0, 3, 1, 2)  # (B, 8, 1280, 1)
        x_strain = self.encoder_strain(x2).permute(0, 3, 1, 2)

        batch_size, num_nodes, feat_num, feat_size = x_acc.shape
        feats_acc = x_acc.reshape(batch_size * num_nodes, feat_num, feat_size)  # (B*8, 1280, 1)
        feats_strain = x_strain.reshape(batch_size * num_nodes, feat_num, feat_size)

        g_batched = dgl.batch([g] * batch_size)
        g_batched.ndata["acc"] = feats_acc.flatten(1)  # (B*8, 1280)
        g_batched.ndata["strain"] = feats_strain.flatten(1)

        # Acceleration branch
        ha = self.gat1_a(g_batched, g_batched.ndata["acc"]).flatten(1)  # (B*8, 1024)
        g_out_a = self.gat3_a(g_batched, ha).flatten(1)  # (B*8, 512)

        # Strain branch
        hs = self.gat1_s(g_batched, g_batched.ndata["strain"]).flatten(1)
        g_out_s = self.gat3_s(g_batched, hs).flatten(1)

        g_out = torch.cat([g_out_a, g_out_s], dim=1)  # (B*8, 1024)

        gf = self.graph_readout(g_batched, g_out)  # (B, 1024)
        return self.linear(gf)


class CNN(ModelRegGNN):
    """CNN baseline: the same encoder, pooled over time and sensors, followed by a linear head."""

    def __init__(self, num_class=4, in_channel=1):
        nn.Module.__init__(self)
        self.num_class = num_class
        self.graph = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Feature extractor: (B, 1, 500, 8) -> (B, 1280, 1, 1)
        self.encoder = yolo_encoder(in_channel, pool=(16, 8))
        self.linear = nn.Linear(1280, num_class)

    def forward(self, x):
        feat = self.encoder(x).flatten(1)  # (B, 1280)
        return self.linear(feat)


class FEGNN(ModelRegGNN):
    """
    Feature-engineering + GNN baseline.

    Input: (B, 1, featnum, 8) statistics per node (see utils/feateng.py), passed through two
    GATv2 layers (8 heads each), averaged over the graph and mapped to the counts.
    """

    def __init__(self, featnum=8, graph="ladder", num_class=4, in_channel=1):
        nn.Module.__init__(self)
        self.num_class = num_class
        self.graph = graph
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.gat1 = GATv2Conv(in_feats=featnum, out_feats=featnum, num_heads=8, activation=nn.SiLU())
        self.gat2 = GATv2Conv(in_feats=featnum * 8, out_feats=featnum * 8, num_heads=8, activation=nn.SiLU())

        self.graph_readout = AvgPooling()
        self.linear = nn.Linear(featnum * 64, num_class)

    def forward(self, x, g):
        batch_size, _, num_feat, num_nodes = x.shape  # (B, 1, num_feat, 8)

        g_batched = dgl.batch([g] * batch_size)
        g_in = x.squeeze(1).permute(0, 2, 1).reshape(batch_size * num_nodes, num_feat)  # (B*8, num_feat)

        h = self.gat1(g_batched, g_in).flatten(1)  # (B*8, num_feat * 8)
        g_out = self.gat2(g_batched, h).flatten(1)  # (B*8, num_feat * 64)

        gf = self.graph_readout(g_batched, g_out)  # (B, num_feat * 64)
        return self.linear(gf)

    def _preprocess(self, signal_tensor):
        """Raw windows are converted to hand-crafted features before inference."""
        return feature_engineer(signal_tensor)


class FEMLP(FEGNN):
    """Feature-engineering + MLP baseline: the flattened (featnum x 8) statistics go through a 3-layer MLP."""

    def __init__(self, featnum=8, graph=None, num_class=4, hidden_dim=128, dropout=0.2):
        nn.Module.__init__(self)
        self.num_class = num_class
        self.graph = graph
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.mlp = nn.Sequential(
            nn.Linear(featnum * 8, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, num_class),
        )

    def forward(self, x):
        bz, _, num_feat, num_node = x.shape
        x = x.view(bz, -1)  # (B, num_feat * num_node)
        return self.mlp(x)  # (B, num_cls)
