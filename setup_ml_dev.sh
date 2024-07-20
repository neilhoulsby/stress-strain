#!/bin/bash

# Check if NVIDIA drivers are installed on Windows
if ! powershell.exe "Get-WmiObject Win32_VideoController | Where-Object { \$_.Name -match 'NVIDIA' }" > /dev/null 2>&1; then
    echo "NVIDIA drivers are not detected on your Windows system. Please install the latest NVIDIA drivers."
    exit 1
fi

# Function to check if conda environment exists
conda_env_exists() {
    conda info --envs | grep -q "^$1 "
}

# Check if ml_dev environment already exists
if conda_env_exists "ml_dev"; then
    echo "The 'ml_dev' environment already exists."
    read -p "Do you want to use the existing environment? (y/n): " use_existing
    if [[ $use_existing == "y" ]]; then
        env_name="ml_dev"
    else
        read -p "Enter a new name for the environment: " env_name
        while conda_env_exists "$env_name"; do
            echo "Environment '$env_name' already exists. Please choose a different name."
            read -p "Enter a new name for the environment: " env_name
        done
        conda create -n $env_name python=3.9 -y
    fi
else
    env_name="ml_dev"
    conda create -n $env_name python=3.9 -y
fi

# Activate the environment
eval "$(conda shell.bash hook)"
conda activate $env_name

# Install CUDA Toolkit for WSL
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# Install pip packages
pip install --no-cache-dir \
    tensorflow \
    matplotlib \
    flax

# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install JAX with CUDA support
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Set up environment variables
echo "export PATH=\"/usr/local/cuda-11.8/bin:\$PATH\"" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\"/usr/local/cuda-11.8/lib64:\$LD_LIBRARY_PATH\"" >> ~/.bashrc

# Source the updated bashrc
source ~/.bashrc

# Create a Python script to check for GPUs and JAX devices
cat << EOF > check_gpu_and_jax.py
import torch
import jax
import tensorflow as tf

print("PyTorch GPU Check:")
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda if torch.cuda.is_available() else "N/A")
print("Number of GPUs:", torch.cuda.device_count())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPUs detected by PyTorch")

print("\nJAX Device Check:")
print("JAX version:", jax.__version__)
print("Available devices:", jax.devices())
print("Default device:", jax.default_backend())

if len(jax.devices()) > 1 or jax.default_backend() != 'cpu':
    print("JAX is using GPU")
else:
    print("JAX is only detecting CPU")

print("\nTensorFlow GPU Check:")
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("GPU Details:", tf.config.list_physical_devices('GPU'))

# Additional CUDA checks
import subprocess

def run_command(command):
    try:
        output = subprocess.check_output(command, shell=True, universal_newlines=True)
        return output.strip()
    except subprocess.CalledProcessError:
        return "Command failed"

print("\nAdditional CUDA Checks:")
print("nvcc version:", run_command("nvcc --version"))
print("CUDA libraries:", run_command("ldconfig -p | grep cuda"))
EOF

# Run the GPU and JAX check script
python check_gpu_and_jax.py

# Print confirmation and troubleshooting steps
echo "Environment '$env_name' has been set up and packages have been installed."
echo "GPU and JAX availability have been checked (see output above)."

echo "
If CUDA is still not detected, try the following troubleshooting steps:
1. Ensure you have the latest NVIDIA drivers installed on your Windows host system.
2. Verify that you're using WSL 2: wsl --status
3. Check if the NVIDIA CUDA driver is loaded in WSL: nvidia-smi
4. If the CUDA driver is not loaded, you may need to restart your WSL instance or your Windows machine.
5. Make sure your Windows GPU drivers support WSL 2 GPU acceleration.
6. Verify that GPU support is enabled in WSL: wsl --status
"

# Provide instructions for others to reproduce the environment
echo "
To reproduce this environment on another WSL 2 machine:
1. Ensure Conda is installed in your WSL environment.
2. Make sure you have the latest NVIDIA drivers installed on your Windows host system.
3. Run this script to set up the environment and install necessary packages.
4. After installation, you may need to restart your WSL instance or Windows machine.
5. Run: python check_gpu_and_jax.py (to verify GPU and JAX availability)
"
