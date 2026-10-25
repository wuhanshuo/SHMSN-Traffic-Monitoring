# SPDX-License-Identifier: AGPL-3.0-only
"""
Regression metrics used during training and the hourly evaluation reported in the paper.

Class order everywhere: [LeftCar, RightCar, LeftTruck, RightTruck]
(Left = overtaking lane, Right = slow lane, Car = light vehicle, Truck = heavy vehicle).
"""

import torch
import numpy as np
from pathlib import Path
from glob import glob

WINDOWS_PER_HOUR = 720  # 5 s windows


def coeff_determination_numpy(y_pred, y_true, p=None):
    """Per-column R2 (or adjusted R2 for `p` predictors) of (samples, classes) numpy arrays."""
    y_true_mean = np.mean(y_true, axis=0)
    total_sum_of_squares = np.sum((y_true - y_true_mean) ** 2, axis=0)
    residual_sum_of_squares = np.sum((y_true - y_pred) ** 2, axis=0)

    # R2 is undefined (NaN) for a constant target
    r2 = np.where(total_sum_of_squares != 0,
                  1 - (residual_sum_of_squares / total_sum_of_squares),
                  np.nan)

    if p is None:
        return r2
    else:
        n = y_true.shape[0]
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        return adjusted_r2


def coeff_determination(y_pred, y_true):
    """Per-class R2 of (batch, classes) tensors; NaN for classes that are constant in the batch."""
    y_true_mean = torch.mean(y_true, axis=0)
    total_sum_of_squares = torch.sum((y_true - y_true_mean) ** 2, axis=0)
    residual_sum_of_squares = torch.sum((y_true - y_pred) ** 2, axis=0)

    r2 = torch.where(total_sum_of_squares != 0,
                     1 - (residual_sum_of_squares / total_sum_of_squares),
                     torch.tensor(float('nan')).to(y_pred.device))

    return r2  # (num_classes,)


def l1_loss_detach(y_pred, y_true):
    """Per-class sum of absolute errors over the batch."""
    return torch.sum(torch.abs(y_pred - y_true), axis=0)


def l1_loss_detach_np(y_pred, y_true):
    """Numpy version of l1_loss_detach."""
    return np.sum(np.abs(y_pred - y_true), axis=0)


def mse_loss_detach(y_pred, y_true):
    """Per-class sum of squared errors over the batch."""
    return torch.sum((y_pred - y_true) ** 2, axis=0)


def _load_predictions(run):
    """Concatenate all <run>/Test/*_norm.npy predictions and *_gt.npy labels to (4, windows)."""
    run_folder = Path(run) / "Test"
    pred_path_list = glob(str(run_folder / "*_norm.npy"))
    pred_arr = np.concatenate([np.load(path).T for path in sorted(pred_path_list)], axis=1)

    gt_path_list = glob(str(run_folder / "*_gt.npy"))
    gt_arr = np.concatenate([np.load(path).T for path in sorted(gt_path_list)], axis=1)
    return pred_arr, gt_arr


def _mae_accuracy(pred, gt):
    """
    MAE, relative MAE (% of mean ground truth) and accuracy along axis 1.

    Accuracy = 100 * sum(min(pred, gt)) / sum(max(pred, gt)), i.e. the overlap of the
    predicted and true traffic-volume curves (100 % = identical curves).
    """
    mae = np.mean(np.abs(pred - gt), axis=1)
    rmae = 100 * mae / np.mean(gt, axis=1)
    accuracy = 100 * np.sum(np.minimum(pred, gt), axis=1) / np.sum(np.maximum(pred, gt), axis=1)
    return mae, rmae, accuracy


def eval_2cls(run, LOGGER=None):
    """
    Hourly metrics for the two aggregated classes:
    light vehicles (both lanes) and heavy vehicles in the right (slow) lane.
    Heavy vehicles in the left lane are too rare (~1 per mille of samples) to be evaluated.
    """
    pred_arr, gt_arr = _load_predictions(run)

    # Hourly aggregation
    pred_hour = pred_arr.reshape(4, -1, WINDOWS_PER_HOUR).sum(axis=2)
    gt_hour = gt_arr.reshape(4, -1, WINDOWS_PER_HOUR).sum(axis=2)

    pred_light = np.sum(pred_hour[[0, 1], :], axis=0, keepdims=True)
    gt_light = np.sum(gt_hour[[0, 1], :], axis=0, keepdims=True)

    pred_heavy = pred_hour[[3], :]
    gt_heavy = gt_hour[[3], :]

    pred_to_compute = np.concatenate([pred_light, pred_heavy], axis=0)  # (2, hours)
    gt_to_compute = np.concatenate([gt_light, gt_heavy], axis=0)  # (2, hours)

    mae, rmae, accuracy = _mae_accuracy(pred_to_compute, gt_to_compute)
    if LOGGER:
        LOGGER.info(f"2Cls MAE: {mae}, Relative MAE: {rmae}, Accuracy: {accuracy}")
    else:
        print(f"2Cls MAE: {mae}, Relative MAE: {rmae}, Accuracy: {accuracy}")


def eval_4cls(run, compute="hour", LOGGER=None):
    """
    Metrics for the four classes, either per 5 s window (compute="raw") or per hour
    (compute="hour", as reported in Tables 4 and 5 of the paper).
    """
    pred_arr, gt_arr = _load_predictions(run)

    # Hourly aggregation
    pred_hour = pred_arr.reshape(4, -1, WINDOWS_PER_HOUR).sum(axis=2)
    gt_hour = gt_arr.reshape(4, -1, WINDOWS_PER_HOUR).sum(axis=2)

    compute_dict = {
        "raw": (pred_arr, gt_arr),
        "hour": (pred_hour, gt_hour),
    }
    pred_to_compute, gt_to_compute = compute_dict[compute]  # (4, windows) or (4, hours)

    mae, rmae, accuracy = _mae_accuracy(pred_to_compute, gt_to_compute)
    if LOGGER:
        LOGGER.info(f"4Cls MAE: {mae}, Relative MAE: {rmae}, Accuracy: {accuracy}")
    else:
        print(f"4Cls MAE: {mae}, Relative MAE: {rmae}, Accuracy: {accuracy}")
