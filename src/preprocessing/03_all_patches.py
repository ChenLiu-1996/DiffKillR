'''
Patchify the entire dataset.
'''

import cv2
import os
import numpy as np
from tqdm import tqdm


def patchify_and_save(image, patches_folder, patch_size):

    for h_idx in tqdm(range(image.shape[0] // patch_size)):
        for w_idx in range(image.shape[1] // patch_size):

            h_loc = h_idx * patch_size
            w_loc = w_idx * patch_size

            patch_image_path = '%s/image/H%d_W%d_patch_%dx%d.png' % (
                patches_folder, h_loc, w_loc, patch_size, patch_size)
            os.makedirs(os.path.dirname(patch_image_path), exist_ok=True)

            h_begin = h_loc
            w_begin = w_loc
            h_end = min(h_begin + patch_size, image.shape[0])
            w_end = min(w_begin + patch_size, image.shape[1])

            patch_image = image[h_begin:h_end, w_begin:w_end, :]

            # NOTE: This is a hot fix. Not sure what is a good heuristic.
            # Only save patches with significant BLUE channel.
            if patch_image[..., 2].max() < 200:
                continue
            # If it's all dark, ignore.
            if patch_image.mean() < 50:
                continue
            # If it's all bright, also ignore.
            if patch_image.mean() > 200:
                continue

            # Handle edge cases: literally on the edge.
            if patch_image.shape != (patch_size, patch_size, 3):
                h_diff = patch_size - patch_image.shape[0]
                w_diff = patch_size - patch_image.shape[1]
                patch_image = np.pad(patch_image,
                                        pad_width=((0, h_diff), (0, w_diff), (0, 0)),
                                        mode='constant')

            patch_image = cv2.cvtColor(patch_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(patch_image_path, patch_image)

    return


if __name__ == '__main__':
    patch_size = 32
    image_path = '../../raw_data/A28-87_CP_lvl1_HandE_1_Merged_RAW_ch00.tif'

    patches_folder = '../../data/A28-87_CP_lvl1_HandE_1_Merged_RAW_ch00_patch_%dx%d/' % (
        patch_size, patch_size)

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    assert len(image.shape) == 3
    assert image.shape[-1] == 3

    # Divide the image and label into patches.
    patchify_and_save(image, patches_folder, patch_size)
