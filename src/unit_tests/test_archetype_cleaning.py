import os
import sys
import cv2
from matplotlib import pyplot as plt

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir + '/utils/')
from archetype_cleaning import archetype_remove_background


if __name__ == '__main__':
    patch_id_list = [
        'TCGA-A7-A13E-01Z-00-DX1_H0W408_patch_96x96',
        'TCGA-A7-A13E-01Z-00-DX1_H70W286_patch_96x96',
        'TCGA-E2-A14V-01Z-00-DX1_H904W271_patch_96x96',
        'TCGA-E2-A14V-01Z-00-DX1_H904W376_patch_96x96',
    ]

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 15
    fig = plt.figure(figsize=(12, 12))

    for row_idx, patch_id in enumerate(patch_id_list):
        image_path = f'../../data/MoNuSeg/MoNuSegByCancer_patch_96x96/Breast/train/images/{patch_id}.png'
        label_path = f'../../data/MoNuSeg/MoNuSegByCancer_patch_96x96/Breast/train/labels/{patch_id}.png'
        image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

        image, background_image, inpainted_image, cleaned_image = \
            archetype_remove_background(image, label, return_intermediates=True)

        ax = fig.add_subplot(4, 4, 1 + 4 * row_idx)
        ax.imshow(image)
        ax.set_axis_off()
        if row_idx == 0:
            ax.set_title('Original Image')
        ax = fig.add_subplot(4, 4, 2 + 4 * row_idx)
        ax.imshow(background_image)
        ax.set_axis_off()
        if row_idx == 0:
            ax.set_title('Background Image')
        ax = fig.add_subplot(4, 4, 3 + 4 * row_idx)
        ax.imshow(inpainted_image)
        ax.set_axis_off()
        if row_idx == 0:
            ax.set_title('Inpainted Image')
        ax = fig.add_subplot(4, 4, 4 + 4 * row_idx)
        ax.imshow(cleaned_image)
        ax.set_axis_off()
        if row_idx == 0:
            ax.set_title('Final Cleaned Image')
        fig.tight_layout(pad=2)
        fig.savefig('archetype_cleaning.png')
