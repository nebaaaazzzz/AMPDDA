# AutoMP-DDA

AutoMP-DDA is a multi-relational graph neural network for drug-disease association prediction. It leverages Fast Graph Transformer Networks (FastGTNs) to automatically learn optimal metapaths in heterogeneous information networks, improving the prediction of potential drug-disease associations.

## Overview

This repository contains the implementation of AMPDDA, which integrates Graph Transformer Networks (GTNs) to capture complex semantic relationships between drugs and diseases. The model is designed to handle large-scale heterogeneous graphs efficiently.

## Requirements

The project requires Python 3.12 and the following key libraries:

- PyTorch 
- DGL
- PyG (Torch Geometric)
- NumPy
- Pandas
- Scikit-learn
- Torch Scatter
- Torch Sparse

Please refer to `requirements.txt` for the specific versions of all dependencies.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/nebaaaazzzz/AutoMP-DDA.git
    cd AutoMP-DDA
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

    *Note: The `requirements.txt` file specifies CPU-only versions for PyTorch and related geometric libraries (`torch==2.4.0+cpu`, `torch-scatter`, `torch-sparse`). Ensure your environment is compatible or modify the file for GPU support if needed.*

## Usage

To train the model, run `main.py` with the desired arguments.

### Basic Usage

```bash
python main.py --dataset Cdataset
```

### Command Line Arguments

The following arguments can be used to configure the training process:

**General Arguments:**
- `-da`, `--dataset`: Dataset to use. Choices: `['Cdataset', 'Kdataset', 'Bdataset', 'Fdataset']`. Default: `Cdataset`.
- `-sp`, `--saved_path`: Path to save training results. Default: `result/<current_time>`.
- `-se`, `--seed`: Global random seed. Default: `42`.

**Training Arguments:**
- `-fo`, `--nfold`: Number of folds for K-fold cross-validation. Default: `10`.
- `-ep`, `--epoch`: Number of epochs for training. Default: `3000`.
- `-lr`, `--learning_rate`: Learning rate. Default: `0.01`.
- `-wd`, `--weight_decay`: Weight decay. Default: `0.0`.
- `-pa`, `--patience`: Early stopping patience. Default: `300`.

**Model Arguments:**
- `-hf`, `--hidden_feats`: Dimension of hidden tensor in the model. Default: `128`.
- `-he`, `--num_heads`: Number of attention heads. Default: `5`.
- `-dp`, `--dropout`: Dropout rate. Default: `0.4`.

**FastGTN Arguments:**
- `--num_layers`: Number of layers in FastGTN unit. Default: `1`.
- `--num_FastGTN_layers`: Total number of FastGTN layers. Default: `1`.
- `--num_channels`: Number of channels. Default: `1`.
- `--non_local`: Enable non-local graph construction inside FastGTN. Default: `False`.
- `--non_local_weight`: Initial weight to assign to non-local channel if enabled. Default: `0.0`.
- `--K`: Top-K for non-local neighbor selection in FastGTN. Default: `8`.
- `--beta`: Mixing weight between original and learned features in FastGTN. Default: `0.5`.
- `--channel_agg`: How to aggregate channels in FastGTN. Choices: `['concat', 'mean']`. Default: `mean`.
- `--remove_self_loops`: If set, remove self loops when normalizing in FastGTN GCNConv. Default: `False`.

### Example

Train on `Cdataset` with 5-fold cross-validation, 128 hidden features, and learning rate 0.005:

```bash
python main.py --dataset Cdataset --nfold 5 --hidden_feats 128 --learning_rate 0.005
```

## File Structure

- `dataset/`: Contains the datasets used for training and evaluation.
- `main.py`: Main entry point for the application.
- `model.py`: Defines the AutoMP-DDA model architecture.
- `fast_gtn.py`: Implementation of FastGTNs for automatic metapath learning.
- `train.py`: Contains the training loop and evaluation logic.
- `decoders.py`: Decoder modules for link prediction.
- `load_data.py`: Utilities for loading and processing graph data.
- `utils.py`: General utility functions.
- `args.py`: Argument parsing configuration.
- `requirements.txt`: List of Python dependencies.

## Citation

If you use this code for your research, please cite our work:

