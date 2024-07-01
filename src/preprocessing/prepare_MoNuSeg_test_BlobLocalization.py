'''
For the test set of MoNuSeg,
Instead of using the ground truth localization of nuclei,
We use an off-the-shelf method to detect the nuclei.
'''

import cv2
import os
import numpy as np
from tqdm import tqdm
from glob import glob
from simple_nuclei_detector import detect_nuclei
#from simple_nuclei_detector_cl import detect_nuclei


def patchify_and_save(image, image_id, centroid_list,
                      patches_folder, patch_size):
    '''
    Divide the image into patches and save them.

    image: original image.
    image_id: id of the image. This should be unique.
    centroid_list: list of centroids for each polygon/cell.

    '''
    for centroid in tqdm(centroid_list, desc='patchifying...'):
        centroid[0], centroid[1] = int(centroid[0]), int(centroid[1])

        patch_image_path = '%s/image/%s_H%d_W%d_patch_%dx%d.png' % (
            patches_folder, image_id, centroid[0] - patch_size // 2,
            centroid[1] - patch_size // 2, patch_size, patch_size)
        os.makedirs(os.path.dirname(patch_image_path), exist_ok=True)

        h_begin = max(centroid[0] - patch_size // 2, 0)
        w_begin = max(centroid[1] - patch_size // 2, 0)
        h_end = min(h_begin + patch_size, image.shape[0])
        w_end = min(w_begin + patch_size, image.shape[1])

        patch_image = image[h_begin:h_end, w_begin:w_end, :]

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


def process_MoNuSeg_Testdata(patch_size=32):
    folder = '../../external_data/MoNuSeg/MoNuSegTestData'

    annotation_files = sorted(glob(f'{folder}/*.xml'))
    image_files = sorted(glob(f'{folder}/*.tif'))

    # delete the folder if it already exists
    patch_size = 32 # NOTE: Should be the same as the aug patch data.
    patches_folder = '../../data/MoNuSegTestData_BlobLocalization_patch_%dx%d/' % (patch_size, patch_size)
    if os.path.exists(patches_folder):
        os.system(f'rm -r {patches_folder}')
    os.makedirs(patches_folder)

    all_centroids_list = []
    for i, annotation_file in enumerate(tqdm(annotation_files)):
        image_id = os.path.basename(annotation_file).split('.')[0]
        image_file = f'{folder}/{image_id}.tif'
        if image_file not in image_files:
            print(f'Image file {image_file} not found.')
            continue

        image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        assert len(image.shape) == 3
        assert image.shape[-1] == 3

        # Use blob detection to get the centroids.
        centroids_list = detect_nuclei(image)
        all_centroids_list.extend(centroids_list)

        # Divide the image into patches.
        patchify_and_save(image, image_id, centroids_list, patches_folder, patch_size)

    print('Done processing all images and annotations: annotated cells: %d' % len(all_centroids_list))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--aug_patch_size', type=int, default=32)
    args = parser.parse_args()

    aug_patch_size = args.aug_patch_size
    process_MoNuSeg_Testdata(aug_patch_size)

