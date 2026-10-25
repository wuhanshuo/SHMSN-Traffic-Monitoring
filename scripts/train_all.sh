#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Train and test every model of the paper. Run from the repository root, either directly
# (bash scripts/train_all.sh) or as a SLURM job (sbatch scripts/train_all.sh).
# Adjust the resources below and activate your Python environment before submitting.

#SBATCH --job-name=shmsn-traffic
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --mem-per-cpu=16g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -e
mkdir -p slurm_logs
start_time=$(date +%s)
python -V
nvidia-smi || true

# Proposed model: CNN + GNN
python -m train.single --channel acc
python -m train.single --channel strain
python -m train.dual

# Baselines
python -m train.cnn --channel acc
python -m train.cnn --channel strain
python -m tools.feat_write --data data/train --out data/train_feat   # features for FE-GNN / FE-MLP
python -m train.fegnn --channel acc
python -m train.fegnn --channel strain
python -m train.femlp --channel acc
python -m train.femlp --channel strain

echo "Total execution time: $(($(date +%s) - start_time)) seconds"
