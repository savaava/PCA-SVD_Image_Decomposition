# PCA-SVD Image Decomposition

Python script that loads JPEG images from the `Images/` folder, assembles an
**N × p dataset matrix** (N = number of images, p = flattened pixel count), and
applies **PCA** and **SVD** on it.

## Requirements

```
pip install numpy matplotlib pillow
```

## Usage

```bash
python PCA_images.py
```

The script:

1. **Builds the N × p matrix** – every JPEG in `Images/` is converted to
   grayscale, resized to 64 × 64, and flattened into a row vector.
2. **PCA via SVD** – centres the matrix and decomposes it with
   `numpy.linalg.svd`; prints the variance explained by each principal
   component.
3. **SVD image compression** – applies SVD directly to the full-resolution
   grayscale image, reconstructs it with k = 5, 20, 50, 100 singular values,
   and saves a comparison figure to `Images/svd_compression.png`.
