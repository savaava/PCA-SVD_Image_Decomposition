import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def load_images_as_dataset(images_dir, target_size=(64, 64)):
    """Load all JPEG images from *images_dir* and return an N×p dataset matrix.

    Each image is converted to grayscale, resized to *target_size*, and
    flattened into a row vector so that the resulting matrix has shape
    (N, p) where N is the number of images and p = target_size[0] *
    target_size[1] is the feature dimensionality.

    Parameters
    ----------
    images_dir  : str            – path to the directory containing JPEG files
    target_size : (int, int)     – (height, width) to resize every image to

    Returns
    -------
    X         : np.ndarray, shape (N, p)
    filenames : list[str]
    """
    filenames = sorted(
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg"))
    )
    rows = []
    for fname in filenames:
        path = os.path.join(images_dir, fname)
        img = Image.open(path).convert("L")        # grayscale
        img = img.resize((target_size[1], target_size[0]))  # PIL: (W, H)
        rows.append(np.array(img, dtype=np.float64).flatten())

    X = np.array(rows)   # shape (N, p)
    return X, filenames


def pca_via_svd(X, n_components=None):
    """Perform PCA on the N×p dataset matrix *X* using SVD.

    The data are mean-centred before decomposition.

    Parameters
    ----------
    X            : np.ndarray, shape (N, p) – dataset matrix
    n_components : int | None               – components to keep (default: all)

    Returns
    -------
    X_proj                  : np.ndarray (N, k) – data projected onto PCs
    components              : np.ndarray (k, p) – principal components (rows)
    explained_variance_ratio: np.ndarray (k,)
    mean                    : np.ndarray (p,)   – per-feature mean
    """
    mean = X.mean(axis=0)
    X_centered = X - mean

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    explained_variance = S ** 2 / (X.shape[0] - 1) if X.shape[0] > 1 else S ** 2
    total = explained_variance.sum()
    evr = explained_variance / total if total > 0 else np.zeros_like(explained_variance)

    if n_components is None:
        n_components = len(S)

    components = Vt[:n_components]
    X_proj = X_centered @ components.T

    return X_proj, components, evr[:n_components], mean


def svd_image_compression(image_array, k):
    """Reconstruct a grayscale image keeping only the top *k* singular values.

    Parameters
    ----------
    image_array : np.ndarray (H, W) – grayscale image (float)
    k           : int               – number of singular values to retain

    Returns
    -------
    reconstructed     : np.ndarray (H, W) – compressed image clipped to [0, 255]
    compression_ratio : float             – original / compressed element count
    """
    k = min(k, min(image_array.shape))
    U, S, Vt = np.linalg.svd(image_array, full_matrices=False)

    reconstructed = (U[:, :k] * S[:k]) @ Vt[:k, :]
    reconstructed = np.clip(reconstructed, 0, 255)

    original_elements = image_array.shape[0] * image_array.shape[1]
    compressed_elements = k * (image_array.shape[0] + image_array.shape[1] + 1)
    compression_ratio = original_elements / compressed_elements

    return reconstructed, compression_ratio


if __name__ == "__main__":
    images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Images")

    # ── 1. Build the N×p dataset matrix ──────────────────────────────────────
    target_size = (64, 64)
    X, filenames = load_images_as_dataset(images_dir, target_size=target_size)
    N, p = X.shape
    print(f"Dataset matrix X shape: {N} observations × {p} features")
    print(f"Images loaded: {filenames}")

    # ── 2. PCA via SVD on the N×p matrix ─────────────────────────────────────
    n_components = min(N, p, 10)
    X_proj, components, evr, mean = pca_via_svd(X, n_components=n_components)
    print(f"\nPCA – top {n_components} component(s):")
    for i, r in enumerate(evr):
        print(f"  PC{i + 1}: {r * 100:.2f}% variance explained")
    print(f"  Projected data shape: {X_proj.shape}")

    # ── 3. SVD compression of the first image ────────────────────────────────
    first_img_path = os.path.join(images_dir, filenames[0])
    img_gray = np.array(
        Image.open(first_img_path).convert("L"), dtype=np.float64
    )
    print(f"\nSVD compression on '{filenames[0]}' "
          f"({img_gray.shape[0]}×{img_gray.shape[1]}):")

    k_values = [5, 20, 50, 100]
    fig, axes = plt.subplots(1, len(k_values) + 1, figsize=(16, 4))
    axes[0].imshow(img_gray, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    for ax, k in zip(axes[1:], k_values):
        recon, ratio = svd_image_compression(img_gray, k)
        ax.imshow(recon, cmap="gray")
        ax.set_title(f"k={k}\nratio={ratio:.1f}×")
        ax.axis("off")
        print(f"  k={k:3d}: compression ratio = {ratio:.2f}×")

    plt.tight_layout()
    out_path = os.path.join(images_dir, "svd_compression.png")
    plt.savefig(out_path, dpi=80)
    print(f"\nFigure saved → {out_path}")
    plt.show()
