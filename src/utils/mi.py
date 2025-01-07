"""
Mutual Information for images.
"""

import numpy as np
from sklearn.metrics import mutual_info_score
from scipy import ndimage


def mutual_information(hist_2d):
    """Compute mutual information from a joint histogram."""
    # Convert the 2D histogram to probabilities
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)  # marginal for x over y
    py = np.sum(pxy, axis=0)  # marginal for y over x
    px_py = np.outer(px, py)  # product of marginals

    # Compute the MI (sum of pxy * log(pxy / (px * py)))
    nzs = pxy > 0  # Only consider non-zero values
    mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
    return mi

# Example usage with two images
def MI_image(image1, image2, bins=32):
    """Compute the mutual information between two images."""
    # Compute a 2D histogram of both images
    hist_2d, _, _ = np.histogram2d(image1.ravel(), image2.ravel(), bins=bins)
    return mutual_information(hist_2d)

def MI_batched_image(image1_batched, image2_batched, bins=32):
    """Compute the mutual information between two batched images."""
    mi_arr = []
    for batch_idx in range(image1_batched.shape[0]):
        mi_arr.append(MI_image(image1_batched[batch_idx], image2_batched[batch_idx]))
    mi_arr = np.array(mi_arr)
    return mi_arr

if __name__ == '__main__':
    image = (np.clip(np.random.normal(loc=0, scale=1, size=(256, 256, 3)), -1, 1) + 1) / 2
    image_identical = image.copy()
    image_different = (np.clip(np.random.normal(loc=0, scale=1, size=(256, 256, 3)), -1, 1) + 1) / 2
    mi_identical = MI_image(image, image_identical)
    mi_different = MI_image(image, image_different)
    print('MI for identical images: ', mi_identical, '\nMI for different images: ', mi_different)
