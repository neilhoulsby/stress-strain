#!/bin/bash

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch

# Start Jupyter Notebook
jupyter notebook

# Deactivate the environment when Jupyter is closed
conda deactivate
