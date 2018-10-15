""" Plots original and reconstructed images side by side for comparison.

Author: Tim Silhan
"""

import matplotlib.pyplot as plt

def plot_comparison(original_images, reconstructed_images, save_path):
    """ Plot the original and reconstructed images """
    fig = plt.figure(figsize=(8, 5))
    fig.add_subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(original_images, origin="upper", cmap="gray")
    plt.axis("off")

    fig.add_subplot(1, 2, 2)
    plt.title("Reconstructed")
    plt.imshow(reconstructed_images, origin="upper", cmap="gray")
    plt.axis("off")
    plt.show()

    save = input("Do you want to save the plot? (y/n): ")
    if save == "y":
        print("Saving plot to ", save_path)
        fig.savefig(save_path, bbox_inches='tight')
