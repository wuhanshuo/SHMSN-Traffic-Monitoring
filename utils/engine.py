# SPDX-License-Identifier: AGPL-3.0-only
"""Sensor-graph construction, multi-precision test loop, uncertainty-weighted loss and training plots."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import time
import csv

import torch
import torch.nn as nn

from .metrics import eval_4cls, eval_2cls

import dgl


def create_complete_graph(num_nodes):
    """Fully connected directed graph without self-loops."""
    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edges.append((i, j))

    src, dst = zip(*edges)
    src = torch.tensor(src, dtype=torch.int64)
    dst = torch.tensor(dst, dtype=torch.int64)

    return dgl.graph((src, dst))


def create_cycle(num_nodes):
    """Bidirectional ring 0-1-...-(n-1)-0."""
    src = torch.arange(0, num_nodes)
    dst = torch.cat((torch.arange(1, num_nodes), torch.tensor([0])))

    g = dgl.graph((src, dst))
    g = dgl.to_bidirected(g)

    return g


def create_chopsticks(num_nodes):
    """Two disconnected bidirectional chains: 0-...-(n/2-1) and n/2-...-(n-1)."""
    assert num_nodes % 2 == 0, f"To create a ladder the number of nodes has to be even! Got {num_nodes}"

    half_n = num_nodes // 2
    # First side
    src_1 = torch.arange(half_n - 1)
    dst_1 = torch.arange(1, half_n)

    # Second side
    src_2 = torch.arange(half_n, num_nodes - 1)
    dst_2 = torch.arange(half_n + 1, num_nodes)

    src = torch.cat([src_1, src_2])
    dst = torch.cat([dst_1, dst_2])

    g = dgl.graph((src, dst))
    g = dgl.to_bidirected(g)

    return g


def create_ladder(num_nodes):
    """
    Ladder graph used in the paper.

    The sensors are ordered along one girder (nodes 0 .. n/2-1) and back along the other
    girder (nodes n/2 .. n-1). Node k is connected to its index neighbours k-1 and k+1 (the
    rails) and to its mirror node n-1-k on the opposite girder (the rungs). All edges are
    added in both directions.

    Args:
        num_nodes (int): has to be even, so that there are num_nodes/2 rungs

    Returns:
        dgl.DGLGraph: The constructed graph.
    """
    assert num_nodes % 2 == 0, f"To create a ladder the number of nodes has to be even! Got {num_nodes}"

    g = dgl.graph([])
    g.add_nodes(num_nodes)

    src = []
    dst = []

    for k in range(num_nodes):
        # Rail: neighbour k-1
        if k - 1 >= 0:
            src.append(k)
            dst.append(k - 1)
        # Rail: neighbour k+1
        if k + 1 < num_nodes:
            src.append(k)
            dst.append(k + 1)
        # Rung: mirror node on the other girder
        mirror_k = num_nodes - k - 1
        if mirror_k != k:  # Avoid self-loops
            src.append(k)
            dst.append(mirror_k)

    g.add_edges(torch.tensor(src), torch.tensor(dst))

    return g


def create_graph(num_nodes, edge_type="complete"):
    """Build the sensor graph of type 'complete', 'ladder', 'cycle' or 'chopsticks'."""
    if edge_type == "complete":
        return create_complete_graph(num_nodes)
    elif edge_type == "ladder":
        return create_ladder(num_nodes)
    elif edge_type == "cycle":
        return create_cycle(num_nodes)
    elif edge_type == "chopsticks":
        return create_chopsticks(num_nodes)
    else:
        raise ValueError(f'Graph type is not valid! Expected "complete", "cycle", "chopsticks" or "ladder", got {edge_type}.')


def run_multi_precision_test(model, test_list, channel, base_save_dir, LOGGER, precisions=("fp32", "fp16", "int8")):
    """
    Run model.predict_long() over every signal in `test_list` once per requested
    precision, then run eval_4cls/eval_2cls on each precision's results.

    Each precision writes to its own subfolder (base_save_dir/<precision>/Test/...)
    so the *_norm.npy / *_gt.npy glob used by eval_4cls/eval_2cls does not mix
    predictions across precisions.

    Returns: {precision: {"mae": np.ndarray, "time_sec": float, "n": int}}
    """
    if LOGGER is None:
        import logging
        LOGGER = logging.getLogger(__name__)

    base_save_dir = Path(base_save_dir)
    summary = {}

    for precision in precisions:
        LOGGER.info(f"=== Running inference in {precision.upper()} precision ===")
        precision_dir = base_save_dir / precision

        test_mae_list = []
        start_time = time.time()
        for test_signal in test_list:
            LOGGER.info(f"[{precision}] Testing {Path(test_signal).stem}...")
            _, test_mae = model.predict_long(
                test_signal,
                save_dir=precision_dir / "Test",
                channel=channel,
                LOGGER=LOGGER,
                precision=precision,
            )
            test_mae_list.append(test_mae)
        elapsed = time.time() - start_time

        if not test_mae_list:
            LOGGER.error(f"[{precision}] No test signal was found under the given test_list, skipping.")
            continue

        global_mae = np.mean(np.array(test_mae_list), axis=0)
        LOGGER.info(f"[{precision}] Total MAE on the testset is: {global_mae}")
        LOGGER.info(
            f"[{precision}] Total inference time on {len(test_mae_list)} signals: "
            f"{elapsed:.3f} sec ({elapsed / len(test_mae_list):.4f} sec/signal)"
        )

        eval_4cls(run=precision_dir, LOGGER=LOGGER)
        eval_2cls(run=precision_dir, LOGGER=LOGGER)

        summary[precision] = {"mae": global_mae, "time_sec": elapsed, "n": len(test_mae_list)}

    LOGGER.info("=== Precision comparison summary (global MAE / total time) ===")
    for precision, res in summary.items():
        LOGGER.info(f"[{precision}] MAE={res['mae']}, time={res['time_sec']:.3f}s over {res['n']} signals")

    return summary


class MultiTaskLoss(nn.Module):
    """
    Homoscedastic-uncertainty weighting of per-class losses.

    Kendall, Gal & Cipolla (2018), "Multi-Task Learning Using Uncertainty to Weigh Losses
    for Scene Geometry and Semantics", https://arxiv.org/abs/1705.07115
    """
    def __init__(self, num_tasks=4):
        super().__init__()
        # Learn log(sigma^2) so that sigma^2 = exp(.) stays positive
        self.log_sigma = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, loss_vec):
        """
        loss_vec: Tensor of shape [num_tasks], one loss per class
        """
        sigma_sq = torch.exp(self.log_sigma)
        weighted_loss = loss_vec / (2 * sigma_sq) + self.log_sigma / 2
        return weighted_loss.sum()


def plot_loss(train_loss_list, val_loss_list, criterion_name, save_dir):
    save_dir = Path(save_dir)

    plt.figure(figsize=(6, 4))
    plt.plot(train_loss_list, label='Train Loss')
    plt.plot(val_loss_list, label='Val Loss')
    y_max = min(max(val_loss_list + train_loss_list), 1.0)
    plt.ylim(0, y_max)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve ({criterion_name})')
    plt.legend()
    plt.savefig(save_dir / "loss_curve.png")


def plot_r2(train_r2, val_r2, save_dir):
    """R2 curves of the first two outputs (LeftCar, RightCar)."""
    fig, axs = plt.subplots(2, 1, figsize=(6, 8))
    x_axis = range(len(train_r2))

    axs[0].plot(x_axis, [a[0] for a in train_r2], label='Train R2', marker='.')
    axs[0].plot(x_axis, [a[0] for a in val_r2], label='Val R2', marker='.')
    axs[0].set_title('LeftCar/R2')

    axs[1].plot(x_axis, [a[1] for a in train_r2], label='Train R2', marker='.')
    axs[1].plot(x_axis, [a[1] for a in val_r2], label='Val R2', marker='.')
    axs[1].set_title('RightCar/R2')

    for ax in axs.flat:
        ax.set(xlabel='Epoch', ylabel='R2')
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_dir / "R2_curve.png")


def plot_error(metrics_dict, save_dir):
    """Validation MSE and MAE (summed over classes) per epoch."""
    plt.clf()
    plt.figure(figsize=(6, 4))
    mse = [a.sum() for a in metrics_dict["MSE"]]
    mae = [a.sum() for a in metrics_dict["MAE"]]
    plt.plot(mse, label='MSE')
    plt.plot(mae, label='MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title(f'Error on Validation Set')
    plt.legend()
    plt.savefig(save_dir / "error_curve.png")


def plot_lr(lr_list, save_dir):
    plt.figure(figsize=(6, 4))
    plt.plot(lr_list, label='lr')
    plt.xlabel('Epoch')
    plt.ylabel('lr')
    plt.title(f'Learning Rate Curve')
    plt.legend()
    plt.savefig(save_dir / "lr_curve.png")


def save_csv(lr_list, train_loss_list, val_loss_list, metrics_dict, train_r2, val_r2, save_dir):
    """Write the per-epoch training history to save_dir/result.csv."""
    with open(save_dir / "result.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "lr",
                         "Train Loss", "Val Loss",
                         "Val MAE", "Val MSE",
                         "Val MAE (Detached)", "Val MSE (Detached)",
                         "Train R2", "Val R2"])
        num_epoch = len(train_loss_list)
        for e in range(num_epoch):
            writer.writerow([e, lr_list[e],
                             train_loss_list[e], val_loss_list[e],
                             metrics_dict['MAE'][e].sum(), metrics_dict['MSE'][e].sum(),
                             metrics_dict['MAE'][e], metrics_dict['MSE'][e],
                             train_r2[e], val_r2[e]
                             ])


if __name__ == "__main__":
    g = create_ladder(8)
    print(g.edges())
