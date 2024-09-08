import cv2
import os
import re
import numpy as np
from glob import glob
from tqdm import tqdm


def extract_h_w(file_path):
    h_w_string = re.findall('H\d+W\d+', file_path)
    assert len(h_w_string) == 1
    h_w_string = h_w_string[0]
    h = int(h_w_string.split('H')[1].split('W')[0])
    w = int(h_w_string.split('W')[1])
    return h, w, h_w_string


if __name__ == '__main__':

    imsize = 200

    for folder in [
        'MoNuSegByCancer_200x200',
        'MoNuSegByCancer_intraimage5pct_200x200',
        'MoNuSegByCancer_intraimage20pct_200x200',
        'MoNuSegByCancer_intraimage50pct_200x200',
        'GLySACByTumor_200x200',
        'GLySACByTumor_intraimage5pct_200x200',
        'GLySACByTumor_intraimage20pct_200x200',
        'GLySACByTumor_intraimage50pct_200x200',
    ]:

        directory_list = sorted(glob('../results/%s/*/' % folder))
        for directory in tqdm(directory_list):
            for model in ['UNet', 'nnUNet', 'MedT', 'PSM', 'LACSS', 'SAM', 'SAM2', 'SAM_Med2D', 'MedSAM']:
                for seed in range(1, 4):

                    if model in ['LACSS', 'SAM', 'SAM2', 'SAM_Med2D', 'MedSAM']:
                        if seed > 1:
                            continue
                        source_folder = '%s/%s/' % (directory, model)
                        stitched_folder = '%s/%s_stitched/' % (directory, model)
                    else:
                        source_folder = '%s/%s_seed%d/' % (directory, model, seed)
                        stitched_folder = '%s/%s_seed%d_stitched/' % (directory, model, seed)

                    os.makedirs(stitched_folder, exist_ok=True)

                    mask_list = sorted(glob(source_folder + '*.png'))

                    base_mask_list = []
                    for mask_path in mask_list:
                        h, w, h_w_string = extract_h_w(mask_path)
                        base_mask_path = mask_path.replace('_' + h_w_string, '')

                        if base_mask_path not in base_mask_list:
                            base_mask_list.append(base_mask_path)

                    for base_mask_path in base_mask_list:
                        mask_patch_list = [item for item in mask_list if base_mask_path.replace('.png', '_') in item]

                        max_h, max_w = 0, 0
                        for mask_patch_path in mask_patch_list:
                            h, w, h_w_string = extract_h_w(mask_patch_path)
                            max_h = max(h, max_h)
                            max_w = max(w, max_w)

                        size_h, size_w = max_h + imsize, max_w + imsize

                        mask_stitched = np.zeros((size_h, size_w))
                        for mask_patch_path in mask_patch_list:
                            h, w, h_w_string = extract_h_w(mask_patch_path)
                            mask_patch = cv2.imread(mask_patch_path, cv2.IMREAD_GRAYSCALE)
                            assert mask_stitched[h : h+imsize, w : w+imsize].sum() == 0
                            mask_stitched[h : h+imsize, w : w+imsize] = mask_patch

                        save_path = base_mask_path.replace(source_folder, stitched_folder)
                        cv2.imwrite(save_path, mask_stitched)
