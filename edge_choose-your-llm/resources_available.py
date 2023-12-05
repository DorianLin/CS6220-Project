# pip install psutil gputil torch tensorflow

import subprocess
import sys
import pkg_resources

def install_packages(packages):
    installed_packages = {pkg.key for pkg in pkg_resources.working_set}
    missing_packages = packages - installed_packages
    if missing_packages:
        python_executable = sys.executable
        try:
            subprocess.check_call([python_executable, '-m', 'pip', 'install', *missing_packages])
            print(f"Successfully installed the missing packages: {missing_packages}")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while installing packages: {missing_packages}. Error: {e}")
            sys.exit(1)

# Define the required packages
required_packages = {'psutil', 'gputil'}

# Install missing packages
install_packages(required_packages)

# Now you can safely import the packages
import psutil
import GPUtil
import platform  # No need to check for platform as it's part of the standard library

# Function to convert bytes to a more readable format
def get_size(bytes, suffix="B"):
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor

# Check for PyTorch installation
try:
    import torch
    torch_installed = True
    torch_version = torch.__version__
except ImportError:
    torch_installed = False
    torch_version = "PyTorch not installed"

# Check for TensorFlow installation
try:
    import tensorflow as tf
    tensorflow_installed = True
    tensorflow_version = tf.__version__
except ImportError:
    tensorflow_installed = False
    tensorflow_version = "TensorFlow not installed"

# Print CPU information
print("="*40, "CPU Info", "="*40)
print("Physical cores:", psutil.cpu_count(logical=False))
print("Total cores:", psutil.cpu_count(logical=True))
cpufreq = psutil.cpu_freq()
print(f"Max Frequency: {cpufreq.max:.2f}Mhz")
print(f"Min Frequency: {cpufreq.min:.2f}Mhz")
print(f"Current Frequency: {cpufreq.current:.2f}Mhz")

# Print memory information
print("="*40, "Memory Information", "="*40)
svmem = psutil.virtual_memory()
print(f"Total: {get_size(svmem.total)}")
print(f"Available: {get_size(svmem.available)}")

# Print GPU information
gpus = GPUtil.getGPUs()
if gpus:
    for i, gpu in enumerate(gpus):
        print(f"=*=*=*=*=*=*=*=*=*=* GPU {i} Info *=*=*=*=*=*=*=*=*=*=")
        print(f"ID: {gpu.id}, Name: {gpu.name}")
        print(f"Total memory: {get_size(gpu.memoryTotal)}")
        print(f"Free memory: {get_size(gpu.memoryFree)}")
        print(f"GPU Utilization: {gpu.load*100}%")
else:
    print("No NVIDIA GPU detected")

# Print Python version
print("="*40, "Python Version", "="*40)
print(platform.python_version())

# Print PyTorch version or not installed label
print("="*40, "PyTorch Version", "="*40)
print(torch_version)

# Print TensorFlow version or not installed label
print("="*40, "TensorFlow Version", "="*40)
print(tensorflow_version)
