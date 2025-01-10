import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def load_grayscale_image(image_path):
    """
    Load a grayscale image from the specified path.
    """
    try:
        image = Image.open(image_path).convert("L")
        return np.array(image)

    except FileNotFoundError:
        print("The file does not exist.")
        return None


def display_images(original, reconstructed):
    """
    Display the original and reconstructed images side by side.
    to compare the quality of the reconstructed image.
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Reconstructed Image")
    plt.imshow(reconstructed, cmap="gray")
    plt.axis("off")
    plt.show()



# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def initialize_codebook(blocks, codebook_size):
    """
    Initialize the codebook by selecting the first codebook_size unique blocks from the dataset.
    """
    unique_blocks = np.unique(blocks, axis=0)
    if len(unique_blocks) < codebook_size:
        raise ValueError("Not enough unique blocks to initialize the codebook.")
    return unique_blocks[:codebook_size]



def assign_blocks_to_clusters(blocks, codebook):
    """ Assign each block to the closest cluster centroid and calculate the total error."""
    labels = []
    total_error = 0
    for block in blocks:
        distances = np.linalg.norm(codebook - block, axis=1)
        closest_idx = np.argmin(distances)
        labels.append(closest_idx)
        total_error += distances[closest_idx] ** 2
    return np.array(labels), total_error


def update_codebook(blocks, labels, codebook_size):
    new_codebook = []
    for i in range(codebook_size):
        cluster_blocks = blocks[labels == i]
        if len(cluster_blocks) > 0:
            # Average blocks to compute new centroid
            new_codebook.append(np.mean(cluster_blocks, axis=0))
        else:
            # Reinitialize empty clusters with random blocks
            new_codebook.append(blocks[np.random.randint(len(blocks))])
    return np.array(new_codebook)


def manual_vector_quantization(image, vector_size, codebook_size, max_iterations=10, tolerance=1e-4):
    """Compress the image using manual vector quantization."""
    height, width = image.shape

    # Pad the image if dimensions are not divisible by vector_size
    padded_height = (height + vector_size[0] - 1) // vector_size[0] * vector_size[0]
    padded_width = (width + vector_size[1] - 1) // vector_size[1] * vector_size[1]
    padded_image = np.zeros((padded_height, padded_width), dtype=image.dtype)
    padded_image[:height, :width] = image

    # Divide the image into vectors (blocks)
    blocks = []
    for i in range(0, padded_height, vector_size[0]):
        for j in range(0, padded_width, vector_size[1]):
            block = padded_image[i:i + vector_size[0], j:j + vector_size[1]].flatten()
            blocks.append(block)
    blocks = np.array(blocks)

    # Initialize the codebook manually
    codebook = initialize_codebook(blocks, codebook_size)

    # Iteratively update the codebook and assign blocks
    for iteration in range(max_iterations):
        labels, error = assign_blocks_to_clusters(blocks, codebook)
        new_codebook = update_codebook(blocks, labels, codebook_size)

        if np.linalg.norm(new_codebook - codebook) < tolerance:
            print(f"Converged in {iteration + 1} iterations with error {error:.2f}")
            break

        codebook = new_codebook

    # Encode the image (find the closest codebook vector for each block)
    compressed_image = labels.reshape((padded_height // vector_size[0], padded_width // vector_size[1]))

    return compressed_image, codebook, (padded_height, padded_width)


def vector_quantization_decompression(compressed_image, codebook, vector_size, original_shape):
    """Reconstruct the image from the compressed image and the codebook."""
    padded_height, padded_width = compressed_image.shape[0] * vector_size[0], compressed_image.shape[1] * vector_size[1]
    decompressed_image = np.zeros((padded_height, padded_width), dtype=codebook.dtype)

    for i in range(compressed_image.shape[0]):
        for j in range(compressed_image.shape[1]):
            idx = compressed_image[i, j]
            if idx >= len(codebook):
                raise ValueError(f"Index {idx} in compressed_image exceeds codebook size {len(codebook)}")
            block = codebook[idx].reshape(vector_size)
            decompressed_image[i * vector_size[0]:(i + 1) * vector_size[0], j * vector_size[1]:(j + 1) * vector_size[1]] = block

    # Crop to original image dimensions
    return decompressed_image[:original_shape[0], :original_shape[1]]



image_path = "./id_Grayscale_vs_Black_White_vs_Monochrome_01.jpg"  # Replace with your grayscale image path
vector_size = (4, 4)  # Block size (4x4 pixels)
codebook_size = 16    # Number of codebook vectors

# Input/Output
image = load_grayscale_image(image_path)

if image is not None:
    # Compression
    compressed, codebook, padded_shape = manual_vector_quantization(image, vector_size, codebook_size)

    # Decompression
    reconstructed = vector_quantization_decompression(compressed, codebook, vector_size, image.shape)

    # Display results
    display_images(image, reconstructed)


