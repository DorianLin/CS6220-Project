'''
This is the code to get some important local resource info
CPU: # of Physical cores, # of Total cores, Max/Min/Current CPU frequency (not available for Apple Silicon)
Memory: Total memory, currently available memory, NVIDIA GPU information (if present)

Dependencies:
Package managing tool: pip
No external installation required: subprocess, sys, platform
External Packages: psutil, GPUtil, pkg_resources
Conditionally Imported Packages (Checked for Installation but Not Necessarily Used): torch, tensorflow
'''

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

# Function to convert bytes to a more readable format
def get_size(bytes, is_gpu=False, suffix="B"):
    factor = 1024
    units = ["", "K", "M", "G", "T", "P"] if not is_gpu else ["M", "G", "T", "P"]
    for unit in units:
        if bytes < factor:
            if not suffix == "B":
                return round(bytes, 2)
            else:
                return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


def get_resources_info():
    info_list = []
    # Define the required packages
    required_packages = {'psutil', 'gputil'}

    # Install missing packages
    install_packages(required_packages)

    # Now you can safely import the packages
    import psutil
    import GPUtil
    import platform  # No need to check for platform as it's part of the standard library

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
    physical_cores, total_cores = psutil.cpu_count(logical=False), psutil.cpu_count(logical=True)
    info_list.extend([physical_cores, total_cores])
    print("-"*40, "CPU Info", "-"*40)
    print("Physical cores:", physical_cores)
    print("Total cores:", total_cores)


    # Attempt to get CPU frequency
    # Currently do not support Apple Silicon, see https://github.com/giampaolo/psutil/issues/1892
    # A potential fix: https://github.com/dehydratedpotato/socpowerbud, 
    # code: https://github.com/dehydratedpotato/socpowerbud/blob/41d8e0ae0ec2953f01a4bc30605f9c4df80906bd/osx-cpufreq/main.m#L3-L51
    try:
        cpufreq = psutil.cpu_freq()
        max_freq, min_freq, current_freq = cpufreq.max, cpufreq.min, cpufreq.current
        info_list.extend([max_freq, min_freq, current_freq])
        print(f"Max Frequency: {max_freq:.2f}Mhz")
        print(f"Min Frequency: {min_freq:.2f}Mhz")
        print(f"Current Frequency: {current_freq:.2f}Mhz")
    except FileNotFoundError:
        print("CPU frequency information not available via psutil, trying sysctl...")
        try:
            # Use sysctl to get CPU frequency information
            result = subprocess.check_output(["sysctl", "hw.cpufrequency"], text=True)
            print(result)
            if "hw.cpufrequency:" in result:
                max_freq = int(result.split(":")[1].strip()) / 1_000_000  # Convert from Hz to MHz
                info_list.extend([max_freq, None, None])
                print(f"CPU Frequency (from sysctl): {max_freq:.2f}MHz")
            else:
                print("Unexpected output format from sysctl.")
                info_list.extend([None, None, None])
        except subprocess.CalledProcessError as e:
            print(f"Failed to get CPU frequency information using sysctl: {e}")
            info_list.extend([None, None, None])
        except ValueError:
            print("Invalid value received from sysctl.")
            info_list.extend([None, None, None])
        except IndexError:
            print("Unexpected output format from sysctl, unable to parse frequency.")
            info_list.extend([None, None, None])

    # Print memory information
    print("-"*40, "Memory Information", "-"*40)
    svmem = psutil.virtual_memory()
    total_mem, available_mem = get_size(svmem.total, False, ""), get_size(svmem.available, False, "") # in GB
    info_list.extend([total_mem, available_mem])
    print(f"Total: {get_size(svmem.total)}")
    print(f"Available: {get_size(svmem.available)}")

    # Print GPU information
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu_info = []
        for i, gpu in enumerate(gpus):
            gpu_id, gpu_name, gpu_memory_total, gpu_memory_free, gpu_load = gpu.id, gpu.name, get_size(gpu.memoryTotal, True, ""), get_size(gpu.memoryFree, True, ""), gpu.load
            print(f"-*-*-*-*-*-*-*-*-*-* GPU {i} Info *-*-*-*-*-*-*-*-*-*-")
            print(f"ID: {gpu_id}, Name: {gpu_name}")
            print(f"Total memory: {get_size(gpu.memoryTotal, True)}")
            print(f"Free memory: {get_size(gpu.memoryFree, True)}")
            print(f"GPU Utilization: {gpu_load*100}%")
            gpu_info.append([gpu_id, gpu_name, gpu_memory_total, gpu_memory_free, gpu_load])
        info_list.append(gpu_info)
    else:
        print("No NVIDIA GPU detected")
        info_list.append(None)

    # Print Python version
    print("-"*40, "Python Version", "-"*40)
    py_version = platform.python_version()
    print(py_version)

    # Print PyTorch version or not installed label
    print("-"*40, "PyTorch Version", "-"*40)
    print(torch_version)

    # Print TensorFlow version or not installed label
    print("-"*40, "TensorFlow Version", "-"*40)
    print(tensorflow_version)

    info_list.extend([py_version, torch_version, tensorflow_version])
    return info_list

if __name__ == "__main__":
    info_list = get_resources_info()
    print(info_list)
