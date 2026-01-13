# Installing llama-cpp-python with CUDA Support on Ubuntu 22.04

This guide provides step-by-step instructions for installing llama-cpp-python with CUDA support on Ubuntu 22.04, including common troubleshooting steps.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation Steps](#installation-steps)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### 1. Check CUDA Installation

Verify that CUDA is installed on your system:

```bash
nvcc --version
nvidia-smi
```

### 2. Check Your GPU Model

Identify your GPU to determine the correct CUDA architecture:

```bash
nvidia-smi --query-gpu=name --format=csv,noheader
```

**Common CUDA Compute Capabilities:**
- RTX 40xx series (4060/4070/4080/4090): `89`
- RTX 30xx series (3060/3070/3080/3090): `86`
- RTX 20xx series (2060/2070/2080): `75`
- GTX 16xx series (1660): `75`
- GTX 10xx series (1080/1060): `61`

---

## Installation Steps

### Step 1: Install System Dependencies

```bash
sudo apt update
sudo apt install -y nvidia-cudnn python3-pip python3-dev python3-venv gcc g++ cmake
```

### Step 2: Install or Upgrade CUDA Toolkit (Optional)

If you need to upgrade from CUDA 11.5 to CUDA 12.6:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-6
```

### Step 3: Configure Environment Variables

Add CUDA to your PATH and LD_LIBRARY_PATH:

```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.6
```

Make it permanent by adding to `~/.bashrc`:

```bash
echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda-12.6' >> ~/.bashrc
source ~/.bashrc
```

Verify CUDA version:

```bash
nvcc --version
```

### Step 4: Create Virtual Environment (Recommended)

```bash
python3 -m venv ~/llama-env
source ~/llama-env/bin/activate
```

### Step 5: Upgrade pip and Packaging Tools

```bash
pip install --upgrade pip setuptools wheel packaging
```

### Step 6: Install llama-cpp-python with CUDA Support

**Option A: Auto-detect GPU Architecture (Try this first)**

```bash
CUDACXX=/usr/local/cuda-12.6/bin/nvcc CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=native" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade --verbose
```

**Option B: Manually Specify GPU Architecture (If auto-detect fails)**

Replace `89` with your GPU's compute capability from Step 2:

```bash
CUDACXX=/usr/local/cuda-12.6/bin/nvcc CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=89" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade --verbose
```

---

## Verification

Test that CUDA support is working:

```python
from llama_cpp import Llama

llm = Llama(
    model_path="./model/gemma-3-4b-it-q4_0.gguf", # Example
    n_gpu_layers=-1,  # Offload all layers to GPU
    n_ctx=2048,
    verbose=True
)

# You should see CUDA initialization messages in the output
```

Check GPU usage during inference:

```bash
watch -n 0.5 nvidia-smi
```

---

## Troubleshooting

### Error 1: dpkg was interrupted

**Error Message:**
```
E: dpkg was interrupted, you must manually run 'sudo dpkg --configure -a' to correct the problem
```

**Solution:**

```bash
sudo dpkg --configure -a
sudo apt-get update
sudo apt-get upgrade
```

Wait for the command to complete fully before proceeding.

---

### Error 2: Target Packages Configured Multiple Times

**Error Message:**
```
W: Target Packages (somerville/binary-amd64/Packages) is configured multiple times in /etc/apt/sources.list.d/oem-somerville-cinccino-meta.list:1 and /etc/apt/sources.list.d/oem-somerville-muk-meta.list:1
```

**Cause:** Duplicate OEM repository entries.

**Solution:**

Remove duplicate repository files:

```bash
sudo rm /etc/apt/sources.list.d/oem-somerville-muk-meta.list
sudo rm /etc/apt/sources.list.d/oem-somerville-tentacool-meta.list
sudo apt update
```

---

### Error 3: AttributeError: module 'packaging.utils' has no attribute 'InvalidName'

**Error Message:**
```
AttributeError: module 'packaging.utils' has no attribute 'InvalidName'
error: metadata-generation-failed
```

**Cause:** Outdated `packaging` library incompatible with `scikit-build-core`.

**Solution:**

Use a virtual environment and upgrade packaging tools:

```bash
python3 -m venv ~/llama-env
source ~/llama-env/bin/activate
pip install --upgrade pip setuptools wheel packaging
```

Then proceed with installation.

---

### Error 4: Unsupported gpu architecture 'compute_'

**Error Message:**
```
nvcc fatal : Unsupported gpu architecture 'compute_'
*** CMake configuration failed
```

**Cause:** CMake failed to auto-detect GPU architecture.

**Solution:**

Manually specify your GPU's CUDA compute capability. First, identify your GPU:

```bash
nvidia-smi --query-gpu=name --format=csv,noheader
```

Then install with explicit architecture (e.g., 89 for RTX 4090):

```bash
CUDACXX=/usr/local/cuda-12.6/bin/nvcc CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=89" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade --verbose
```

---

### Error 5: GPU Not Being Used Despite Installation

**Symptoms:** Model runs but GPU utilization is 0%.

**Solution:**

1. Ensure you set `n_gpu_layers` when initializing the model:

```python
llm = Llama(
    model_path="./model.gguf",
    n_gpu_layers=-1,  # -1 means use all available layers
    n_ctx=2048,
    verbose=True
)
```

2. Verify CUDA is in your PATH:

```bash
export PATH="/usr/local/cuda-12.6/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH"
```

3. Check the verbose output for CUDA initialization messages.

---

## Alternative Installation Methods

### Using Pre-built Wheels

If building from source continues to fail, try using pre-built wheels:

```bash
# For CUDA 12.x
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122

# For CUDA 12.1
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

**Note:** Pre-built wheels may not be optimized for your specific GPU architecture.

---

## Additional Notes

- **Virtual Environment Recommended:** Using a virtual environment helps avoid conflicts with system packages.
- **Compilation Time:** Building from source with CUDA support takes several minutes.
- **CUDA Version:** This guide uses CUDA 12.6, but you can adapt it for other CUDA versions by changing the version numbers in paths.
- **Multiple CUDA Versions:** If you have multiple CUDA versions installed, ensure your environment variables point to the version you want to use.

---

## Quick Reference Commands

**Check CUDA version:**
```bash
nvcc --version
```

**Check GPU:**
```bash
nvidia-smi
```

**Activate virtual environment:**
```bash
source ~/llama-env/bin/activate
```

**Uninstall llama-cpp-python:**
```bash
pip uninstall llama-cpp-python -y
```

**Monitor GPU usage:**
```bash
watch -n 0.5 nvidia-smi
```

---

## Resources

- [llama-cpp-python GitHub](https://github.com/abetlen/llama-cpp-python)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [NVIDIA GPU Compute Capabilities](https://developer.nvidia.com/cuda-gpus)

---

**Last Updated:** January 9, 2026
