# Graph-Crossformer-for-Multivariate-Spatial-Temporal-Forecasting

# DIMIGNN: Graph Neural Networks with Diversity-Aware Neighbor Selection and Dynamic Multi-Scale Fusion for Multivariate Time Series Forecasting

This repository contains the research code for our work on multivariate PV and load forecasting with graph neural networks and cross-scale temporal modeling.

The project was developed as part of a collaboration between NYU and USC.  
Codebase organized and maintained by **Guibin (Jeremy) Chen**, with contributions from an MSc student.

## Overview

We study multivariate time series forecasting on distribution-level PV and load data, modeled as a graph of nodes (feeders, buses, or sites).  
Our method, **DIMIGNN**, combines:

- Diversity-aware neighbor selection on the spatial graph,
- Dynamic multi-scale temporal fusion,
- A unified encoder-decoder architecture for long-horizon forecasting.

We benchmark against strong baselines including:
- Transformer-based forecasters (e.g., Crossformer-style models),
- Classical temporal models.

## Repository Structure

- `main_crossformer.py`  
  Training entry point for DIMIGNN and baseline models.

- `eval_crossformer.py`  
  Evaluation and prediction scripts.

- `cross_models/`  
  Model definitions (DIMIGNN and baseline architectures).

- `cross_exp/`  
  Experiment wrappers, argument parsing, and training logic.

- `data/`  
  Generic data loading utilities (`Dataset_MTS`, scaling, splits).

- `utils/`  
  Metrics, time feature encoders, and training utilities.

- `datasets/`  
  Placeholder for preprocessed PV/load datasets.  
  **Raw CSV files are not released** due to data-sharing restrictions.

- `scripts/`  
  Example SLURM or shell scripts for running experiments on clusters.

## Data

We assume multivariate time series stored as per-node CSV files, merged into a consistent format.  
Each dataset should be converted into standardized tensors with shape `(T, N, F)`:

- `T`: total time steps  
- `N`: number of nodes  
- `F`: number of features per node

See `data/data_loader.py` for details on how the `Dataset_MTS` class expects the data.

> Note: The original PV/load CSV files used in our experiments are **not** included in this repository.  
> You may either: (1) request access if appropriate, or (2) adapt the code to your own datasets.

## Running Experiments

Example (single-GPU):

```bash
python main_crossformer.py \
  --model DIMIGNN \
  --root_path ./datasets/ \
  --data_path your_data_config.csv \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --batch_size 32 \
  --learning_rate 1e-4
