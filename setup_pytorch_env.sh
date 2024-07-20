#!/bin/bash

# Set environment name
ENV_NAME="pytorch"

# Create a new conda environment
conda create -n $ENV_NAME python=3.9 -y

# Activate the environment
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Install PyTorch with CUDA support (adjust cuda version if necessary)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other required packages
conda install -y matplotlib numpy typing_extensions

# Export the environment to a YAML file
conda env export --no-builds | grep -v "prefix:" > environment.yml

# Export pip requirements
pip list --format=freeze > requirements.txt

# Verify the installation
python -c "import functools, matplotlib, os, pdb, time, typing, matplotlib.pyplot, numpy, torch, torch.nn, torch.optim; from torch.utils.data import DataLoader, TensorDataset; print('All modules successfully imported!')"

echo "Environment setup complete. You can recreate this environment using:"
echo "conda env create -f environment.yml && conda activate $ENV_NAME && pip install -r requirements.txt"
