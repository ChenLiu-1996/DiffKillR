'''
Patchify the TissueNet dataset for the following purposes.

FACT: Each image exclusively contain cells from the same tissue.

1. Annotate some cells of 1 image, and infer on the remaining cells in the same image.
   Analyze the effect of % annotated cells on seg result.

2. Annotate some cells of 1 image, and infer on another image of same tissue.
   Analyze the effect of % annotated cells.

3. Annotate some cells of 1 image, and infer on another image of different tissue.
   Analyze the effect of % annotated cells.

'''
import cv2
import os
import numpy as np
from typing import List
from tqdm import tqdm
from glob import glob
from matplotlib import pyplot as plt
import seaborn as sns


def patchify_and_save(tissue_types_list,
                      mask_list_by_tissue,
                      patches_folder,
                      patch_size,
                      ratio_list: List[float] = [0.01, 0.1, 0.2, 0.5, 1.0],
                      val_in_train_val: float = 0.2):

    np.random.seed(1)

    #NOTE: Purpose 1. Annotate some cells of 1 image, and infer on the remaining cells in the same image.
    #                 Analyze the effect of % annotated cells on seg result.

    # Get the image with the most number of cells for each tissue type.
    for tissue_type in tqdm(tissue_types_list):
        mask_path_with_most_cell, max_cell_count = None, 0
        for mask_path in mask_list_by_tissue[tissue_type]:
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            unique_cell_ids = set(np.unique(mask)) - set([0])  # don't count background
            cell_count = len(unique_cell_ids)
            if cell_count > max_cell_count:
                mask_path_with_most_cell = mask_path
                max_cell_count = cell_count
        image_path_with_most_cell = mask_path_with_most_cell.replace(MASK_DIR, IMAGE_DIR)
        image_with_most_cell = cv2.cvtColor(cv2.imread(image_path_with_most_cell, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        mask_with_most_cell = cv2.imread(mask_path_with_most_cell, cv2.IMREAD_UNCHANGED)
        del mask, cell_count

        # Sample certain ratios of these cells as train+val, and move the rest to test.
        cell_id_list = np.unique(mask_with_most_cell)
        cell_id_list_final = []
        # Filter out cells that are too big.
        for cell_id in cell_id_list:
            foreground = np.argwhere(mask_with_most_cell == cell_id)
            if (max(foreground[:, 0]) - min(foreground[:, 0]) < patch_size) and \
               (max(foreground[:, 1]) - min(foreground[:, 1]) < patch_size):
                cell_id_list_final.append(cell_id)

        for ratio in ratio_list:
            if ratio == 1:
                # We won't consider this case.
                continue

            # At least 2 for train+val.
            train_val_ids = np.random.choice(cell_id_list_final, size=max(2, int(ratio * max_cell_count)), replace=False)
            # At least 1 for val.
            val_ids = np.random.choice(train_val_ids, size=max(1, int(val_in_train_val * len(train_val_ids))), replace=False)
            train_ids = np.array(list(set(train_val_ids) - set(val_ids)))
            # Remaining are for test.
            test_ids = np.array(list(set(cell_id_list_final) - set(train_ids) - set(val_ids)))

            # Save the images.
            for id_list, split_str in zip([train_ids, val_ids, test_ids], ['train', 'val', 'test']):
                for idx in id_list:
                    centroid = np.argwhere(mask_with_most_cell == idx).sum(0) / (mask_with_most_cell == idx).sum()
                    centroid = [int(item) for item in centroid]
                    h_begin = max(centroid[0] - patch_size // 2, 0)
                    w_begin = max(centroid[1] - patch_size // 2, 0)
                    h_end = min(h_begin + patch_size, image_with_most_cell.shape[0])
                    w_end = min(w_begin + patch_size, image_with_most_cell.shape[1])

                    patch_image = image_with_most_cell[h_begin:h_end, w_begin:w_end, :]
                    patch_mask = (mask_with_most_cell == idx)[h_begin:h_end, w_begin:w_end]

                    # Handle edge cases: literally on the edge.
                    if patch_image.shape != (patch_size, patch_size, 3):
                        h_diff = patch_size - patch_image.shape[0]
                        w_diff = patch_size - patch_image.shape[1]
                        patch_image = np.pad(patch_image,
                                            pad_width=((0, h_diff), (0, w_diff), (0, 0)),
                                            mode='constant')
                        patch_mask = np.pad(patch_mask,
                                            pad_width=((0, h_diff), (0, w_diff)),
                                            mode='constant')

                    patch_image_path = patches_folder + 'same_image_generalization/ratio_%s/%s/%s/images/%s_cell_%s.png' % (
                        ratio, tissue_type, split_str, os.path.basename(mask_path_with_most_cell).replace('.png', ''), str(idx).zfill(5))
                    os.makedirs(os.path.dirname(patch_image_path), exist_ok=True)
                    patch_image = cv2.cvtColor(patch_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(patch_image_path, patch_image)

                    patch_mask_path = patches_folder + 'same_image_generalization/ratio_%s/%s/%s/masks/%s_cell_%s.png' % (
                        ratio, tissue_type, split_str, os.path.basename(mask_path_with_most_cell).replace('.png', ''), str(idx).zfill(5))
                    os.makedirs(os.path.dirname(patch_mask_path), exist_ok=True)
                    cv2.imwrite(patch_mask_path, patch_mask.astype(np.uint8) * 255)

    # #NOTE: Purpose 2. Annotate some cells of 1 image, and infer on another image of same tissue.
    # #                 Analyze the effect of % annotated cells.
    # output_folder = patches_folder + 'same_tissue_generalization/'

    # #NOTE: Purpose 3. Annotate some cells of 1 image, and infer on another image of different tissue.
    # #                 Analyze the effect of % annotated cells.
    # output_folder = patches_folder + 'other_tissue_generalization/'


    # # image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    # # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # # assert len(image.shape) == 3
    # # assert image.shape[-1] == 3

    # for cell_type, centroid_list in class_centroid_map.items():
    #     for centroid in tqdm(centroid_list):
    #         patch_image_path = '%s/image/%s_H%d_W%d_patch_%dx%d.png' % (
    #             patches_folder, cell_type, centroid[0] - patch_size // 2,
    #             centroid[1] - patch_size // 2, patch_size, patch_size)
    #         patch_label_path = '%s/label/%s_H%d_W%d_patch_%dx%d.png' % (
    #             patches_folder, cell_type, centroid[0] - patch_size // 2,
    #             centroid[1] - patch_size // 2, patch_size, patch_size)
    #         patch_colored_label_path = '%s/colored_label/%s_H%d_W%d_patch_%dx%d.png' % (
    #             patches_folder, cell_type, centroid[0] - patch_size // 2,
    #             centroid[1] - patch_size // 2, patch_size, patch_size)
    #         os.makedirs(os.path.dirname(patch_image_path), exist_ok=True)
    #         os.makedirs(os.path.dirname(patch_label_path), exist_ok=True)
    #         os.makedirs(os.path.dirname(patch_colored_label_path),
    #                     exist_ok=True)

    #         h_begin = max(centroid[0] - patch_size // 2, 0)
    #         w_begin = max(centroid[1] - patch_size // 2, 0)
    #         h_end = min(h_begin + patch_size, image.shape[0])
    #         w_end = min(w_begin + patch_size, image.shape[1])

    #         # print('centroid', centroid)
    #         # print('h, w', h_begin, h_end, w_begin, w_end)

    #         patch_image = image[h_begin:h_end, w_begin:w_end, :]
    #         patch_label = label_map[h_begin:h_end, w_begin:w_end]

    #         # Handle edge cases: literally on the edge.
    #         if patch_image.shape != (patch_size, patch_size, 3):
    #             h_diff = patch_size - patch_image.shape[0]
    #             w_diff = patch_size - patch_image.shape[1]
    #             patch_image = np.pad(patch_image,
    #                                  pad_width=((0, h_diff), (0, w_diff), (0,
    #                                                                        0)),
    #                                  mode='constant')
    #             patch_label = np.pad(patch_label,
    #                                  pad_width=((0, h_diff), (0, w_diff)),
    #                                  mode='constant')

    #         patch_image = cv2.cvtColor(patch_image, cv2.COLOR_RGB2BGR)
    #         cv2.imwrite(patch_image_path, patch_image)
    #         cv2.imwrite(patch_label_path, patch_label)

    #         # NOTE: Colored label only used for visual inspection!
    #         patch_label_colored = np.zeros_like(patch_image)
    #         patch_label_colored[patch_label == 1] = (0, 0, 255)
    #         patch_label_colored[patch_label == 2] = (0, 255, 0)
    #         patch_label_colored[patch_label == 3] = (255, 0, 0)
    #         patch_label_colored[patch_label == 4] = (255, 255, 255)
    #         cv2.imwrite(patch_colored_label_path, patch_label_colored)

    return


if __name__ == '__main__':
    patch_size = 32
    plot_distribution = False

    IMAGE_DIR = '../../external_data/TissueNet/images/'
    MASK_DIR = '../../external_data/TissueNet/labels_cytoplasm/'
    patches_folder = '../../external_data/TissueNet/cellseg_partition/'

    image_list = glob(IMAGE_DIR + '*.png')
    tissue_types_list = np.unique([os.path.basename(p).split('_')[1].split('.png')[0] for p in image_list])

    if plot_distribution:
        # Plot the height/width distribution.
        plt.rcParams['font.family'] = 'serif'
        fig_HW_dist = plt.figure(figsize=(len(tissue_types_list) * 0.9, 6))

    image_list_by_tissue = {}
    mask_list_by_tissue = {}
    cell_size_h_list_by_tissue = []
    cell_size_w_list_by_tissue = []
    for tissue_type in tissue_types_list:
        image_list_by_tissue[tissue_type] = glob(IMAGE_DIR + '*%s*.png' % tissue_type)
        mask_list_by_tissue[tissue_type] = glob(MASK_DIR + '*%s*.png' % tissue_type)

        assert len(image_list_by_tissue[tissue_type]) == len(mask_list_by_tissue[tissue_type])

        cell_count = 0
        cell_size_h_list = []
        cell_size_w_list = []
        for mask_path in tqdm(mask_list_by_tissue[tissue_type]):
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            unique_cell_ids = set(np.unique(mask)) - set([0])  # don't count background
            cell_count += len(unique_cell_ids)
            if plot_distribution:
                for cell_id in unique_cell_ids:
                    foreground = np.argwhere(mask == cell_id)
                    cell_size_h_list.append(max(foreground[:, 0]) - min(foreground[:, 0]))
                    cell_size_w_list.append(max(foreground[:, 1]) - min(foreground[:, 1]))

        cell_size_h_list_by_tissue.append(np.array(cell_size_h_list))
        cell_size_w_list_by_tissue.append(np.array(cell_size_w_list))

        print('Tissue type: %s has %d images and %d cells.' % (
            tissue_type, len(image_list_by_tissue[tissue_type]), cell_count))
        if plot_distribution:
            print('max height = %d, max width = %d. height (25%%, 50%%, 75%%) = (%d, %d, %d), width (25%%, 50%%, 75%%) = (%d, %d, %d)' % (
                np.max(cell_size_h_list), np.max(cell_size_w_list),
                np.percentile(cell_size_h_list, 25), np.percentile(cell_size_h_list, 50), np.percentile(cell_size_h_list, 75),
                np.percentile(cell_size_w_list, 25), np.percentile(cell_size_w_list, 50), np.percentile(cell_size_w_list, 75)))

    if plot_distribution:
        fig_HW_dist.suptitle('Distribution of cell heights and widths in pixels', fontsize=15)

        ax = fig_HW_dist.add_subplot(2, 1, 1)
        ax.spines[['right', 'top']].set_visible(False)
        sns.boxplot(cell_size_h_list_by_tissue, ax=ax, width=0.6, showfliers=False)
        ax.axhline(y = patch_size, color = 'k', linestyle = '--')
        ax.set_xticks(np.arange(len(tissue_types_list)))
        ax.set_xticklabels(['\n'.join(item.split(' ')) for item in tissue_types_list])
        ax.set_ylabel('Height', fontsize=12)

        ax = fig_HW_dist.add_subplot(2, 1, 2)
        ax.spines[['right', 'top']].set_visible(False)
        sns.boxplot(cell_size_w_list_by_tissue, ax=ax, width=0.6, showfliers=False)
        ax.axhline(y = patch_size, color = 'k', linestyle = '--')
        ax.set_xticks(np.arange(len(tissue_types_list)))
        ax.set_xticklabels(['\n'.join(item.split(' ')) for item in tissue_types_list])
        ax.set_ylabel('width', fontsize=12)

        fig_HW_dist.tight_layout(pad=2)
        fig_HW_dist.savefig('TissueNet_data_cell_size_distirbution.png')

    del cell_size_h_list_by_tissue, cell_size_w_list_by_tissue

    # Divide the image and label into patches.
    patchify_and_save(tissue_types_list, mask_list_by_tissue, patches_folder, patch_size)
