import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def matrix_to_image(X_reconstructed: np.ndarray, original_shape: tuple):
    H, W, C = original_shape

    # Ritorna alla shape (H, W, C)
    arr = X_reconstructed.reshape(H, W, C)

    # Clip dei valori nell'intervallo [0, 255] e conversione a uint8
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    img_out = Image.fromarray(arr) # crea un'immagine PIL a partire dall'array numpy
    return img_out

def load_image_as_matrix(img_file_path: str):
    img = Image.open(img_file_path).convert("RGB") # assicura 3 canali (RGB)
    img_arr = np.array(img, dtype=np.float64)
    # img_arr[0, 0, 0] -> è il valore del canale R del pixel in alto a sinistra (un singolo numero)
    # img_arr[0, 0, :] -> è il pixel in alto a sinistra con i valori RGB quindi è un array di 3 numeri (R, G, B)
    # img_arr[0, 0, :].flatten() -> è il pixel in alto a sinistra con i valori RGB appiattiti quindi è un array di 3 numeri (R, G, B)

    H, W, C = img_arr.shape 
    # H: altezza dell'immagine
    # W: larghezza dell'immagine
    # C: numero di canali (C=3 per RGB) 

    # Ogni riga dell'immagine diventa un'osservazione con W*C feature
    X = np.zeros((H, W * C), dtype=np.float64)  # shape: (H, W*C)
    for i in range(H):
        X[i, :] = img_arr[i, :, :].flatten()  # appiattisce la riga i in un array di W*C elementi
#        idx_X = 0
#        for ii in range(W):
#            X[i, idx_X] = img_arr[i, ii, 0]  # canale R
#            X[i, idx_X + 1] = img_arr[i, ii, 1]  # canale G
#            X[i, idx_X + 2] = img_arr[i, ii, 2]  # canale B
#            idx_X += 3

    return X, (H, W, C)

def alter_image_gaussian_noise(X: np.ndarray, deviation: float = 50.0):
    X_out = X.astype(float) # copia in float per evitare overflow/underflow con uint8
    
    noise = np.random.normal(0, deviation, X.shape) # matrice di rumore inizializzata a zero

    X_out += noise
    return np.clip(X_out, 0, 255).astype(np.uint8)

def alter_image_uniform_noise(X: np.ndarray, alteration_percentage: float = 0.50):
    X_out = X.astype(float)

    noise = (
        np.random.randint(-255, 256, X.shape) *
        (np.random.rand(X.shape[0], X.shape[1]) < alteration_percentage))

    X_out += noise

    return np.clip(X_out, 0, 255).astype(np.uint8)


def load_all_imgs():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(current_dir, "Images")

    imgs_data = {}
    # - key: nome del file dell'immagine
    # - value: (nome del file dell'immagine, dataset X dell'immagine, original_shape dell'immagine)
    for file_name in os.listdir(images_dir):
        if file_name.endswith(".jpeg") or file_name.endswith(".jpg") or file_name.endswith(".png"):
            img_file_path = os.path.join(images_dir, file_name)
            print(f"Caricamento immagine: {file_name}...")
            X, original_shape = load_image_as_matrix(img_file_path)
            imgs_data[file_name] = (file_name, X, original_shape)
    
    return imgs_data

if __name__ == "__main__":
    np.random.seed(42)

    imgs_data = load_all_imgs()

    for (img_name, X, original_shape) in imgs_data.values():
        # ad ogni iterazione, plot delle due immagini: quella originale e quella alterata
        plt.figure()
        print(f"Alterazione immagine: {img_name}...")
        X_altered1 = alter_image_uniform_noise(X)
        X_altered2 = alter_image_gaussian_noise(X)
        img_original_out = matrix_to_image(X, original_shape)
        img_altered1_out = matrix_to_image(X_altered1, original_shape)
        img_altered2_out = matrix_to_image(X_altered2, original_shape)

        plt.subplot(1, 3, 1)
        plt.imshow(img_original_out)
        plt.title(f"Original Image: {img_name}")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title(f"Altered Image (Uniform Noise): {img_name}")
        plt.imshow(img_altered1_out)
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title(f"Altered Image (Gaussian Noise): {img_name}")
        plt.imshow(img_altered2_out)
        plt.axis('off')

    plt.tight_layout()
    plt.show()