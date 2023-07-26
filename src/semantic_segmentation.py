from typing import Union, Tuple

import numpy as np
from skimage.filters import threshold_otsu
import cv2

import matplotlib.pyplot as plt

def apply_otsu_thresholding(img: Union[str, np.ndarray], nbins:int = 256) -> Tuple[np.ndarray, float]:
    """
    Perform Otsu thresholding on a given image.

    Parameters
    ----------
    img : str or np.ndarray
        Path to image, or BGR image. Need to convert to grayscale first.

    Returns
    -------
    np.ndarray
        Thresholded image.
    float
        Threshold value.
    """
    if isinstance(img, str):
        image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif isinstance(img, np.ndarray):
        if img.shape[-1] != 3:
            raise ValueError('img must be BGR image')
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise TypeError('img must be str or np.ndarray')
    
    thresh = threshold_otsu(image, nbins=nbins)
    binarized_image = image < thresh # NOTE: cells are dark, background is bright, so we want to keep the dark pixels.
    
    return binarized_image, thresh


if __name__ == '__main__':
    # Plot image and binarized segmentation side by side

    # image_path = '../data/NSCLC_S20-8466_03-01_HandE_H1792W9472.jpg'
    # otsu_thresholding(image_path)

    image_path = '../data/NSCLC_S20-8466_03-01_HandE_H1792W9728.jpg'
    binarized_image, _ = apply_otsu_thresholding(image_path)

    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    ax[0].imshow(cv2.imread(image_path, cv2.IMREAD_UNCHANGED))
    ax[0].set_title('Original image')
    ax[1].imshow(cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2GRAY), cmap='gray')
    ax[1].set_title('Grayscale image')
    ax[2].imshow(binarized_image)
    ax[2].set_title('Binarized image')

    plt.show()