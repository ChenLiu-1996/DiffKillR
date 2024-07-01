import os
import cv2
import numpy as np
import skimage.feature
import skimage.segmentation


def detect_nuclei(img: np.array, return_overlay: bool = False):
    if img.shape[-1] == 1:
        # (H, W, 1) to (H, W, 3)
        img = np.repeat(img, 3, axis=-1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Pad white the image to avoid border effects
    gray = cv2.copyMakeBorder(gray, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # Setup SimpleBlobDetector parameters.
    print('Default parameters: =====')
    params = cv2.SimpleBlobDetector_Params()
    for p in dir(params):
        if not p.startswith('__'):
            print(p, getattr(params, p))

    params.minThreshold = 5
    params.maxThreshold = 220

    # params.filterByArea = True
    params.minArea = 150
    params.maxArea = 10000.0

    # params.filterByCircularity = False
    # params.filterByConvexity = False
    # params.filterByInertia = False
    params.minConvexity = 0.8 #0.9499
    params.minDistBetweenBlobs = 1

    # # Create a detector with the parameters
    # detector = cv2.SimpleBlobDetector_create(params)
    print('Updated parameters: =====')
    for p in dir(params):
        if not p.startswith('__'):
            print(p, getattr(params, p))
    detector = cv2.SimpleBlobDetector_create(params)

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
    test_file = '../raw_data/A28-87_CP_lvl1_HandE_1_Merged_RAW_ch00.tif'

    image = cv2.imread(test_file, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    nuclei_list, im_with_keypoints = detect_nuclei(image, return_overlay=True)
    
    print('Number of detected nuclei: ', len(nuclei_list))
    cv2.imwrite('../data/nuclei_detected.png', im_with_keypoints)

    # Save some percentage of the detected nuclei as patches
    patch_size = 32
    percent = 0.1
    sampled_idxs = np.random.choice(np.arange(len(nuclei_list)), int(percent * len(nuclei_list)), replace=False)
    save_dir = f'../data/A28-87_CP_lvl1_HandE_1_Merged_RAW_ch00_test_patch_{patch_size}x{patch_size}_{percent}/'
    os.makedirs(save_dir, exist_ok=True)

    print(f'Patching {len(sampled_idxs)} nuclei...')
    i = 0
    for idx in sampled_idxs:
        h, w = nuclei_list[idx]
        h_start = int(h - patch_size / 2)
        w_start = int(w - patch_size / 2)
        patch = image[h_start:h_start + patch_size, w_start:w_start + patch_size]

        if patch.shape != (patch_size, patch_size, 3):
            #print(f'Patch shape: {patch.shape}')
            diff_w = patch_size - patch.shape[0]
            diff_h = patch_size - patch.shape[1]

            patch = np.pad(patch, ((0, diff_w), (0, diff_h), (0, 0)), mode='constant', constant_values=0)

        # save the patch
        name = f'patch_{i}.png'
        save_path = os.path.join(save_dir, name)
        cv2.imwrite(save_path, patch)
        
        i += 1
    
    print(f'Patched {len(sampled_idxs)} nuclei to {save_dir}')