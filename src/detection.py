from skimage.feature import blob_log

import numpy as np
import matplotlib.pyplot as plt
import cv2

from semantic_segmentation import apply_otsu_thresholding


def detect_blob(img: np.ndarray,
                min_sigma: float = 1,
                max_sigma: float = 50,
                num_sigma: int = 10,
                threshold: float = 0.1,
                overlap: float = 0.5) -> np.ndarray:
    """
    Blob detection using multiscale LoG (Laplacian of Gaussian).

    Parameters
    ----------
    img : np.ndarray
        Input image. Must be grayscale. Blobs are assumed to be light on dark background (white on black).
    min_sigma : float, optional
        Smallest standard deviation of Gaussian kernel, by default 1
    max_sigma : float, optional
        Largest standard deviation of Gaussian kernel, by default 50
    num_sigma : int, optional
        Number of intermediate values of standard deviations to consider between min_sigma and max_sigma, by default 10
    threshold : float, optional
        The absolute lower bound for scale space maxima, by default 0.1
    overlap : float, optional
        A value between 0 and 1. If the area of two blobs overlaps by a fraction greater than threshold, the smaller blob is eliminated, by default 0.5
    
    Returns
    -------
    np.ndarray
        Detected blobs. Each row is a blob (y, x, r) where (y,x) is the center and r is the radius.
    """
    # TODO: max, min sigma can be a sequence of floats for non-isotropic blob detection
    blobs = blob_log(img, min_sigma=min_sigma, max_sigma=max_sigma,
                       num_sigma=num_sigma, threshold=threshold, overlap=overlap)
    
    # radius = sqrt(2) * sigma
    blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
    
    #print(blobs.shape)

    return blobs


if __name__ == '__main__':
    image_path = '../data/NSCLC_S20-8466_03-01_HandE_H1792W9728.jpg'
    binarized_image, _ = apply_otsu_thresholding(image_path)

    blobs = detect_blob(binarized_image, min_sigma=3, max_sigma=50, num_sigma=30, threshold=0.12, overlap=0.5)

    # Plot 
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    og_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    ax[0].imshow(og_image)
    ax[0].set_title('Input')

    ax[1].imshow(binarized_image, cmap='gray')
    ax[1].set_title('Otsu Thresholding')

    ax[2].imshow(binarized_image, cmap='gray')
    ax[2].set_title('LoG blobs')
    ax[2].set_axis_off()
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color='r', linewidth=2, fill=False)
        ax[2].add_patch(c)
        

    plt.tight_layout()
    plt.show()





