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
