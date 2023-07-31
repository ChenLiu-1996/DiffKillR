from typing import Union, Optional

from scipy import ndimage as ndi
import matplotlib.pyplot as plt

import numpy as np
import cv2
from skimage.morphology import disk
from skimage.segmentation import watershed
from skimage import data, color, feature, measure
from skimage.filters import rank
from skimage.util import img_as_ubyte

from detection import detect_blob
from semantic_segmentation import apply_otsu_thresholding


def apply_watershed(img: Union[str, np.ndarray], markers: Optional[np.ndarray], connectivity: int = 1,
                     offset: Optional[np.ndarray] = None, mask: Optional[np.ndarray] = None, 
                     compactness: int = 0, watershed_line: bool = False):
    """
    Apply watershed segmentation to a given image.
    
    Parameters
    ----------
    img : str or np.ndarray Data array where the lowest value points are labeled first. Usually gradients.

    markers: An array marking the basins with the values to be assigned in the label matrix. Zero means not a marker. 
    If None (no markers given), the local minima of the image are used as markers.

    """
    if isinstance(img, str):
        image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    elif isinstance(img, np.ndarray) != True:
        raise TypeError('img must be str or np.ndarray')
    
    labels = watershed(img, markers, connectivity=connectivity, offset=offset, 
                       mask=mask, compactness=compactness, watershed_line=watershed_line)
    
    return labels

if __name__ == '__main__':
    image_path = '../data/NSCLC_S20-8466_03-01_HandE_H1792W9728.jpg'
    save_path = '../output/Segmentation_NSCLC_S20-8466_03-01_HandE_H1792W9728.jpg'
    binarized_image, _ = apply_otsu_thresholding(image_path)

    blobs = detect_blob(binarized_image, min_sigma=3, max_sigma=50, num_sigma=30, threshold=0.12, overlap=0.5)

    # Optional 1: Using detected blobs as markers and binarized image as the image
    print('Using detected blobs as markers and binarized image as the image...')
    print(binarized_image.shape)
    markers = np.zeros_like(binarized_image, dtype=np.uint8)
    print('init ...: ', markers.shape, markers)

    for i in range(len(blobs)):
        blob = blobs[i]
        y, x, radius = blob
        markers[int(y), int(x)] = i
    print('done marking blobs. ', markers.shape, markers, np.sum(markers))

    instance_labels = apply_watershed(
        binarized_image, markers=markers, connectivity=1, offset=None, mask=binarized_image)
    
    # Optional 2: Using detected blobs as markers and the gradient of original image as img
    print('Using detected blobs as markers and the gradient of original image as img...')
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # local gradient (disk(2) is used to keep edges thin)
    gradient = rank.gradient(image, disk(2))
    instance_labels_gradient = apply_watershed(
        gradient, markers=markers, connectivity=1, offset=None, mask=binarized_image)

    
    # Plot 
    print(markers.shape, markers)
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    og_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    ax[0,0].imshow(og_image)
    ax[0,0].set_title('Input')

    ax[0,1].imshow(binarized_image, cmap=plt.cm.gray)
    ax[0,1].set_title('Otsu Thresholding')

    ax[0,2].imshow(color.label2rgb(markers, bg_label=0))
    ax[0,2].set_title('LoG Markers, %d detected' % len(blobs))

    ax[1,0].imshow(og_image, cmap=plt.cm.gray)
    ax[1,0].imshow(instance_labels, cmap=plt.cm.nipy_spectral, alpha=.5)
    ax[1,0].set_title("Binarized + LoG Segmented %d cells" % np.max(instance_labels))

    ax[1,1].imshow(og_image, cmap=plt.cm.gray)
    ax[1,1].imshow(instance_labels_gradient, cmap=plt.cm.nipy_spectral, alpha=.5)
    ax[1,1].set_title("Gradient + LoG Segmented %d cells" % np.max(instance_labels_gradient))

    # Optional 3: Using edit distance
    distance = ndi.distance_transform_edt(binarized_image)

    local_max_coords = feature.peak_local_max(distance, min_distance=7)
    local_max_mask = np.zeros(distance.shape, dtype=bool)
    local_max_mask[tuple(local_max_coords.T)] = True
    markers = measure.label(local_max_mask)

    segmented_cells = apply_watershed(-distance, markers, mask=binarized_image)

    ax[1,2].imshow(og_image, cmap=plt.cm.gray)
    ax[1,2].imshow(segmented_cells, cmap=plt.cm.nipy_spectral, alpha=.5)
    ax[1,2].set_title("Edit Distance Segmented %d cells" % np.max(segmented_cells))

    for a in ax.ravel():
        a.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)

    plt.show()



    

