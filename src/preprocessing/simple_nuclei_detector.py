import cv2
import numpy as np
import skimage.feature
import skimage.segmentation


def detect_nuclei(img: np.array, return_overlay: bool = False):
    if img.shape[-1] == 1:
        # (H, W, 1) to (H, W, 3)
        img = np.repeat(img, 3, axis=-1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detector = cv2.SimpleBlobDetector_create()

    # Detect blobs.
    keypoints = detector.detect(gray)

    nuclei_list = []
    for kp in keypoints:
        (w, h) = kp.pt
        nuclei_list.append([h, w])

    if return_overlay:
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(gray, keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return nuclei_list, im_with_keypoints
    else:
        return nuclei_list


if __name__ == '__main__':
    image_path = '../../external_data/MoNuSeg/MoNuSegByCancer/colon/test/images/TCGA-A6-6782-01A-01-BS1.png'
    mask_path = '../../external_data/MoNuSeg/MoNuSegByCancer/colon/test/masks/TCGA-A6-6782-01A-01-BS1.png'
    image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(18, 6))
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(image)
    ax.set_axis_off()
    ax = fig.add_subplot(1, 3, 2)
    _, pseudomask = detect_nuclei(image, return_overlay=True)
    ax.imshow(pseudomask, cmap='gray')
    ax.set_axis_off()
    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(mask, cmap='gray')
    ax.set_axis_off()
    fig.savefig('check_nuclei_detector.png')
