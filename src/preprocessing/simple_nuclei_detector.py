import cv2
import numpy as np
import skimage.feature
import skimage.segmentation
from skimage.color import rgb2hed


def detect_nuclei(img: np.array, return_overlay: bool = False):
    if img.shape[-1] == 1:
        # (H, W, 1) to (H, W, 3)
        img = np.repeat(img, 3, axis=-1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    # params.minThreshold = 10;
    # params.maxThreshold = 200;

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 100

    # Filter by Circularity
    # params.filterByCircularity = True
    # params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.6

    # Filter by Inertia
    # params.filterByInertia = True
    # params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(gray)

    nuclei_list = []
    for kp in keypoints:
        (w, h) = kp.pt
        nuclei_list.append([h, w])

    if return_overlay:
        pseudomask = cv2.drawKeypoints(np.zeros_like(gray), keypoints, np.array([]), (255, 255, 255),
                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        pseudomask_overlay = cv2.drawKeypoints(gray, keypoints, np.array([]), (0, 0, 255),
                                               cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return nuclei_list, pseudomask, pseudomask_overlay
    else:
        return nuclei_list

def detection_eval(nuclei_list, mask, total_pos):
    # Convert nuclei_list to mask
    mask = (mask > 0) * 1.0
    nuclei_mask = np.zeros_like(mask)
    #print(nuclei_list)
    for n in nuclei_list:
        if n[0] < 0 or n[0] >= mask.shape[0] or n[1] < 0 or n[1] >= mask.shape[1]:
            continue
        nuclei_mask[int(n[0]), int(n[1])] = 1

    # Compute accuracy
    TP = np.sum(nuclei_mask * mask)
    FN = total_pos - TP
    # TN = np.sum((1 - nuclei_mask) * (1 - mask))

    # accuracy = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN)
    # f1 = 2 * precision * recall / (precision + recall)
    return recall, TP, FN


import os
from glob import glob
from Metas import Organ2FileID
from matplotlib import pyplot as plt

def rgb2Hematoxylin(image):
    '''
    Get the Hematoxylin channel from an RGB H&E image.
    '''
    image_HED = rgb2hed(image)

    assert len(image_HED[:, :, 0].shape) == 2
    return image_HED[:, :, 0][:, :, None]


if __name__ == '__main__':
    image_path = '../../external_data/MoNuSeg/MoNuSegByCancer/colon/test/images/TCGA-A6-6782-01A-01-BS1.png'
    mask_path = '../../external_data/MoNuSeg/MoNuSegByCancer/colon/test/masks/TCGA-A6-6782-01A-01-BS1.png'
    image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(image)
    ax.set_axis_off()
    ax = fig.add_subplot(2, 2, 2)
    image_Hematoxylin = rgb2Hematoxylin(image)
    _, pseudomask, pseudomask_overlay = detect_nuclei(image, return_overlay=True)
    ax.imshow(pseudomask_overlay, cmap='gray')
    ax.set_axis_off()
    ax = fig.add_subplot(2, 2, 3)
    ax.imshow(pseudomask, cmap='gray')
    ax.set_axis_off()
    ax = fig.add_subplot(2, 2, 4)
    ax.imshow(mask, cmap='gray')
    ax.set_axis_off()
    fig.savefig('check_nuclei_detector.png')
