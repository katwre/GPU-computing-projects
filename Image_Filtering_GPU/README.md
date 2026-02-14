# Image Filtering with GPU Acceleration

This project demonstrates GPU acceleration for image processing through Gaussian blur filtering. It compares three implementations: CPU baseline, GPU library (CuPy), and custom CUDA kernel.

Image filtering is a fundamental operation in scientific computing. This notebook focuses on Gaussian blur - smoothing images by computing weighted averages of neighboring pixels. The operation is naturally parallel, making it ideal for GPU acceleration.


### Tools

- NumPy - CPU image processing
- CuPy - GPU arrays (NumPy-like API)
- Numba - Custom CUDA kernels


## Setup

```bash
# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Launch notebook
jupyter lab notebooks/ImageFiltering.ipynb
```
