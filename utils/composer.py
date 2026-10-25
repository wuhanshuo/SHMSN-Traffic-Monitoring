# SPDX-License-Identifier: AGPL-3.0-only
"""
Pre-processing of the raw monitoring recordings.

Accelerations (250 Hz) are resampled to the 100 Hz strain rate and both modalities are
stacked into one array (2, T, 8). Accelerations are linearly detrended; strains are
detrended, freed from zero drift with a symmetric EWMA and low-pass filtered at 2.5 Hz.
Each modality is finally divided by a fixed scale (0.02 for acc, 15 for strain).
"""

import pandas as pd
import numpy as np
import scipy.signal as signal
from pathlib import Path

# Exponentially Weighted Moving Average (EWMA) calculation
def ewma_detrend_symmetric(data, alpha=0.1, pad_len=100):
    """
    Bidirectional EWMA with padding to reduce edge effects.

    Args:
        data (np.ndarray): Input data array.
        alpha (float, optional): Smoothing factor for EWMA (default is 0.1).
        pad_len (int, optional): Number of points to pad at start and end (default is 100).

    Returns:
        np.ndarray: Detrended data after applying bidirectional EWMA with padding.
    """
    # Padding: Extend start and end with constant values
    pad_start = np.full(pad_len, data[0])  # Start padding with first value
    pad_end = np.full(pad_len, data[-1])   # End padding with last value
    padded_data = np.concatenate([pad_start, data, pad_end])
    
    # Forward and backward EWMA
    ewma_fwd = np.zeros_like(padded_data)
    ewma_bwd = np.zeros_like(padded_data)
    
    # Forward EWMA calculation
    ewma_fwd[0] = padded_data[0]
    for i in range(1, len(padded_data)):
        ewma_fwd[i] = alpha * padded_data[i] + (1 - alpha) * ewma_fwd[i - 1]
    
    # Backward EWMA calculation
    ewma_bwd[-1] = padded_data[-1]
    for i in range(len(padded_data) - 2, -1, -1):
        ewma_bwd[i] = alpha * padded_data[i] + (1 - alpha) * ewma_bwd[i + 1]
    
    # Average the forward and backward EWMA
    ewma_symmetric = (ewma_fwd + ewma_bwd) / 2
    
    # Remove padding
    detrended_data = (padded_data - ewma_symmetric)[pad_len:-pad_len]
    return detrended_data

# Low-pass filter
def lowpass_filter(data, cutoff=0.1, fs=100.0, order=2):
    """Low-pass Butterworth filter."""
    # Normalize the cutoff frequency
    b, a = signal.butter(order, cutoff / (fs / 2), btype='low')  
    # Use filtfilt for bidirectional filtering
    return signal.filtfilt(b, a, data)

def correct_strain(strain_data, alpha=0.1, cutoff=2.5, fs=100.0):
    """
    Remove the zero drift of strain signals and keep their low-frequency (quasi-static) part.

    Args:
        strain_data (np.ndarray): The raw strain signal, with shape (N, 8), where N is the number of time points and 8 is the number of sensors.
        alpha (float, optional): The smoothing parameter for EWMA (default is 0.1).
        cutoff (float, optional): The cutoff frequency for the low-pass filter (default is 2.5 Hz).
        fs (float, optional): The sampling rate (default is 100.0 Hz).

    Return:
        np.ndarray: The corrected strain signal with zero drift removed and low-frequency components extracted, with shape (N, 8).
    """
    strain_data = np.apply_along_axis(signal.detrend, axis=0, arr=strain_data)  # remove linear trend
    strain_data = np.apply_along_axis(ewma_detrend_symmetric, axis=0, arr=strain_data, alpha=alpha)  # Remove zero drift
    # Keep the low-frequency information
    strain_data = np.apply_along_axis(lowpass_filter, axis=0, arr=strain_data, cutoff=cutoff, fs=fs)
    return strain_data

def concatenate_signals(acc_df, strain_df, norm = True, signal_filter = True):
    """Resample, align, filter and scale the two modalities into one (2, T, 8) array."""
    norm_scale = [0.02, 15]
    # Get acceleration array and downsample to 100Hz
    acc_np = acc_df.to_numpy()
    acc_np = np.expand_dims(acc_np, axis=0)
    acc_np_100hz = signal.resample_poly(acc_np, 100, 250, axis = 1)   # (1, T, 8)

    # Get strain array
    strain_np = strain_df.to_numpy()
    strain_np = np.expand_dims(strain_np, axis=0)

    # Trim the strain data to match the length of acceleration data
    min_length = min(acc_np_100hz.shape[1], strain_np.shape[1])

    # Concatenate the two arrays
    dual_np = np.concatenate((acc_np_100hz[:,:min_length,:], strain_np[:,:min_length,:]), axis=0)

    # filter and detrend
    if signal_filter:
        dual_np[0] = np.apply_along_axis(signal.detrend, axis=0, arr=dual_np[0])  # remove linear trend
        dual_np[1] = correct_strain(dual_np[1], alpha=0.1, cutoff=2.5, fs=100.0)

    if norm:
        for i in range(dual_np.shape[0]):
            dual_np[i] = dual_np[i] / norm_scale[i]

    return dual_np

def read_pickle(fname, norm = True, signal_filter=True):
    """
    Read one raw recording and return (signal (2, T, 8), start timestamp, end timestamp).

    The pickle holds a dict with pandas DataFrames "accelerations" and "strains"; the
    columns listed below select the 8 accelerometers and 8 strain gauges used as graph nodes.
    """
    fname = Path(fname)
    df = pd.read_pickle(fname)

    # Get accelerations
    acc_df = df["accelerations"].loc[:, ["p_1_11", "p_1_13","p_1_15", "p_1_17",
                                        "p_1_7" , "p_1_5" ,"p_1_3" , "p_1_1",]]

    # Get strains
    strain_df = df["strains"].loc[:, ["p_2_4", "p_2_22", "p_2_24", "p_2_14",
                                    "p_2_8", "p_2_20", "p_2_18", "p_2_2",]]
    dual_np = concatenate_signals(acc_df, strain_df, norm, signal_filter)

    start = strain_df.index[0]   # pandas.Timestamp
    end = strain_df.index[-1]

    return dual_np, start, end
