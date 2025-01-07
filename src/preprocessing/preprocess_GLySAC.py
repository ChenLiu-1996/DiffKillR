'''
Read annotations from xml file, find the label maps, and patchify around them.
images are in .tif format, RGB, 1000x1000.

NOTE that we are saving the instance labels (different values for different cells),
and we will only binarize them during data loading. This will be helpful for
distinguishing the target cell from surrounding cells.
'''
import cv2
import os
import numpy as np
from tqdm import tqdm
from glob import glob
from matplotlib import pyplot as plt
import scipy.io
from Metas import GLySAC_Organ2FileID


def load_GLySAC_annotation(mat_path: str) -> list[np.ndarray]:
    '''
    Return the instance label and centroid array.
    '''
    mat = scipy.io.loadmat(mat_path)

    instance_label = mat['inst_map'].astype(np.uint16)  # uint16 is essential!
    centroid_arr = mat['inst_centroid']

    return instance_label, centroid_arr


def process_GLySAC_data():
    '''
    images are in .tif format, RGB, 1000x1000.
    '''

    for subset in ['test', 'train']:

        if subset == 'train':
            image_folder = '../../external_data/GLySAC/Train/Images'
            annotation_folder = '../../external_data/GLySAC/Train/Labels'
        else:
            image_folder = '../../external_data/GLySAC/Test/Images'
            annotation_folder = '../../external_data/GLySAC/Test/Labels'

        out_image_folder = f'../../data/GLySAC/{subset}/images/'
        out_label_folder = f'../../data/GLySAC/{subset}/labels/'
        out_mask_folder = f'../../data/GLySAC/{subset}/masks/'
        out_stats_folder = f'../../data/GLySAC/{subset}/stats/'

        annotation_files = sorted(glob(f'{annotation_folder}/*.mat'))
        image_files = sorted(glob(f'{image_folder}/*.tif'))

        # Size statistics about the annotated cells.
        h_list, w_list = [], []

        for annotation_file in tqdm(annotation_files):
            image_id = os.path.basename(annotation_file).split('.')[0]
            image_file = f'{image_folder}/{image_id}.tif'
            if image_file not in image_files:
                print(f'Image file {image_file} not found.')
                continue

            image = cv2.cvtColor(cv2.imread(image_file, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
            assert len(image.shape) == 3
            assert image.shape[-1] == 3

            # Read the annotation mat.
            label, centroid_arr = load_GLySAC_annotation(annotation_file)
            print('Done reading annotation for image %s' % image_id)
            print('Number of annotated cells: %d' % len(centroid_arr))

            os.makedirs(out_image_folder, exist_ok=True)
            os.makedirs(out_label_folder, exist_ok=True)
            os.makedirs(out_mask_folder, exist_ok=True)

            out_image_path = os.path.join(out_image_folder, image_id + '.png')
            out_label_path = os.path.join(out_label_folder, image_id + '.png')
            out_mask_path = os.path.join(out_mask_folder, image_id + '.png')

            assert np.max(label) < 2**16
            mask = label > 0

            cv2.imwrite(out_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(out_label_path, label)
            cv2.imwrite(out_mask_path, np.uint8(mask * 255))

            # Record the cell sizes.
            for cell_idx in np.unique(label):
                if cell_idx == 0:
                    continue

                rows, cols = np.where(label == cell_idx)
                h_min, h_max = rows.min(), rows.max()
                w_min, w_max = cols.min(), cols.max()

            h_list.append(h_max - h_min + 1)
            w_list.append(w_max - w_min + 1)

        print('GLySAC statistics (%s set)' % subset)
        print('Mean height: %d, Mean width: %d' % (np.mean(h_list), np.mean(w_list)))
        print('Std height: %d, Std width: %d' % (np.std(h_list), np.std(w_list)))

        os.makedirs(out_stats_folder, exist_ok=True)
        stats_fig_save_path = os.path.join(out_stats_folder, 'cell_size_histogram_%s.png' % subset)
        stats_arr_save_path = os.path.join(out_stats_folder, 'cell_sizes_%s.npz' % subset)
        np.savez(stats_arr_save_path, cell_heights=np.array(h_list, dtype=np.float16), cell_widths=np.array(w_list, dtype=np.float16))

        plt.rcParams['font.size'] = 18
        plt.rcParams['font.family'] = 'sans-serif'

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.boxplot([h_list, w_list], widths=0.6)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Cell Height', 'Cell Width'])
        ax.set_ylabel('Pixels', fontsize=24)
        fig.tight_layout(pad=2)
        fig.savefig(stats_fig_save_path, dpi=200)
        plt.close(fig)

    return

def patchify_GLySAC_data_by_tumor_cell_centric(patch_size: int, background_ratio: float = 0.5):
    '''
    images are in .tif format, RGB, 1000x1000.
    '''

    for tumor_type in ['Tumor', 'Normal']:
        for subset in ['test', 'train']:

            if subset == 'train':
                image_folder = '../../external_data/GLySAC/Train/Images'
                annotation_folder = '../../external_data/GLySAC/Train/Labels'
            else:
                image_folder = '../../external_data/GLySAC/Test/Images'
                annotation_folder = '../../external_data/GLySAC/Test/Labels'

            out_image_folder = f'../../data/GLySAC/GLySACByTumor_patch_{patch_size}x{patch_size}/{tumor_type}/{subset}/images/'
            out_label_folder = f'../../data/GLySAC/GLySACByTumor_patch_{patch_size}x{patch_size}/{tumor_type}/{subset}/labels/'
            out_bg_image_folder = f'../../data/GLySAC/GLySACByTumor_patch_{patch_size}x{patch_size}/{tumor_type}/{subset}/background_images/'
            out_stats_folder = f'../../data/GLySAC/GLySACByTumor/{tumor_type}/{subset}/stats/'

            annotation_files = sorted(glob(f'{annotation_folder}/*.mat'))
            image_files = sorted(glob(f'{image_folder}/*.tif'))

            # Size statistics about the annotated cells.
            h_list, w_list = [], []

            for annotation_file in tqdm(annotation_files):
                image_id = os.path.basename(annotation_file).split('.')[0]
                image_file = f'{image_folder}/{image_id}.tif'
                if image_file not in image_files:
                    print(f'Image file {image_file} not found.')
                    continue

                image = cv2.cvtColor(cv2.imread(image_file, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
                assert len(image.shape) == 3
                assert image.shape[-1] == 3
                image_h, image_w = image.shape[:2]

                # Read the annotation mat.
                label, centroid_arr = load_GLySAC_annotation(annotation_file)
                num_cells = len(centroid_arr)
                print('Done reading annotation for image %s' % image_id)
                print('Number of annotated cells: %d' % num_cells)

                os.makedirs(out_image_folder, exist_ok=True)
                os.makedirs(out_label_folder, exist_ok=True)

                out_image_path = os.path.join(out_image_folder, image_id + '.png')
                out_label_path = os.path.join(out_label_folder, image_id + '.png')

                assert np.max(label) < 2**16

                cv2.imwrite(out_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                cv2.imwrite(out_label_path, label)

                # Patchify and save the cell images and labels.
                for coord in centroid_arr:
                    h = int(coord[0] - patch_size // 2)
                    w = int(coord[1] - patch_size // 2)

                    h = min(h, image_h - patch_size)
                    h = max(h, 0)
                    w = min(w, image_w - patch_size)
                    w = max(w, 0)

                    cell_file_name = f'{image_id}_H{h}W{w}_patch_{patch_size}x{patch_size}.png'
                    bg_image_path_to = os.path.join(out_image_folder, cell_file_name)
                    label_path_to = os.path.join(out_label_folder, cell_file_name)
                    os.makedirs(os.path.dirname(bg_image_path_to), exist_ok=True)
                    os.makedirs(os.path.dirname(label_path_to), exist_ok=True)

                    image_patch = image[h : h+patch_size, w : w+patch_size, :]
                    label_patch = label[h : h+patch_size, w : w+patch_size]

                    assert np.max(label_patch) < 2**16

                    cv2.imwrite(bg_image_path_to, cv2.cvtColor(image_patch, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(label_path_to, label_patch)

                # Find background patches that do not contain cells.
                background_locs = find_background_patch_locs(label, patch_size=patch_size)
                num_backgrounds = int(num_cells * background_ratio)
                if len(background_locs) <= 0:
                    print('[Background] No candidate patches found for image %s' % image_id)
                    continue

                np.random.seed(1)
                if num_backgrounds > len(background_locs):
                    background_idx = np.random.choice(len(background_locs), size=num_backgrounds, replace=True)
                else:
                    background_idx = np.random.choice(len(background_locs), size=num_backgrounds, replace=False)
                selected_background_locs = [background_locs[idx] for idx in background_idx]

                # Save background patches.
                for (h, w) in selected_background_locs:
                    bg_file_name = f'{image_id}_H{h}W{w}_patch_{patch_size}x{patch_size}.png'
                    bg_image_patch = image[h : h+patch_size, w : w+patch_size]

                    bg_image_path_to = os.path.join(out_bg_image_folder, bg_file_name)
                    os.makedirs(os.path.dirname(bg_image_path_to), exist_ok=True)
                    cv2.imwrite(bg_image_path_to, cv2.cvtColor(bg_image_patch, cv2.COLOR_RGB2BGR))

                # Record the cell sizes.
                for cell_idx in np.unique(label):
                    if cell_idx == 0:
                        continue

                    rows, cols = np.where(label == cell_idx)
                    h_min, h_max = rows.min(), rows.max()
                    w_min, w_max = cols.min(), cols.max()

                h_list.append(h_max - h_min + 1)
                w_list.append(w_max - w_min + 1)

            print('GLySAC statistics (%s set, %s)' % (subset, tumor_type))
            print('Mean height: %d, Mean width: %d' % (np.mean(h_list), np.mean(w_list)))
            print('Std height: %d, Std width: %d' % (np.std(h_list), np.std(w_list)))

            os.makedirs(out_stats_folder, exist_ok=True)
            stats_fig_save_path = os.path.join(out_stats_folder, 'cell_size_histogram_%s.png' % subset)
            stats_arr_save_path = os.path.join(out_stats_folder, 'cell_sizes_%s.npz' % subset)
            np.savez(stats_arr_save_path, cell_heights=np.array(h_list, dtype=np.float16), cell_widths=np.array(w_list, dtype=np.float16))

            plt.rcParams['font.size'] = 18
            plt.rcParams['font.family'] = 'sans-serif'

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(1, 1, 1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.boxplot([h_list, w_list], widths=0.6)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(['Cell Height', 'Cell Width'])
            ax.set_ylabel('Pixels', fontsize=24)
            fig.tight_layout(pad=2)
            fig.savefig(stats_fig_save_path, dpi=200)
            plt.close(fig)

    return


def find_background_patch_locs(label, patch_size) -> list[str]:
    '''
    TODO: Find background regions given label of the image.
    Args:
        label: np.array of shape (H, W) for instance label
        patch_size: size of the patch (e.g. 96)

    Returns:
        candidate_patches: list of tuple (h, w) recording the top left corner of the background patches
    '''
    # pixel by pixel scan
    stride = 1
    candidate_patches = []
    for h in range(0, label.shape[0], stride):
        for w in range(0, label.shape[1], stride):
            patch_label = label[h:h+patch_size, w:w+patch_size]
            patch_mask = patch_label > 0
            if patch_mask.shape != (patch_size, patch_size):
                continue

            # check if the patch intersects with any label patch
            overlap_threshold = 0
            if np.sum(patch_mask) <= overlap_threshold:
                candidate_patch = (h, w)
                candidate_patches.append(candidate_patch)

    print('[Background] Number of candidate patches: %d' % len(candidate_patches))

    return candidate_patches

def subset_GLySAC_data_by_tumor():
    train_image_folder = '../../data/GLySAC/train/images/'
    train_label_folder = '../../data/GLySAC/train/labels/'
    train_mask_folder = '../../data/GLySAC/train/masks/'
    test_image_folder = '../../data/GLySAC/test/images/'
    test_label_folder = '../../data/GLySAC/test/labels/'
    test_mask_folder = '../../data/GLySAC/test/masks/'

    target_folder = '../../data/GLySAC/GLySACByTumor/'

    for tumor_type in ['Tumor', 'Normal']:
        train_list = GLySAC_Organ2FileID[tumor_type]['train']
        test_list = GLySAC_Organ2FileID[tumor_type]['test']

        for train_item in tqdm(train_list):
            image_path_from = train_image_folder + train_item + '.png'
            label_path_from = train_label_folder + train_item + '.png'
            mask_path_from = train_mask_folder + train_item + '.png'
            image_path_to = os.path.join(target_folder, tumor_type, 'train/images/', train_item + '.png')
            label_path_to = os.path.join(target_folder, tumor_type, 'train/labels/', train_item + '.png')
            mask_path_to = os.path.join(target_folder, tumor_type, 'train/masks/', train_item + '.png')

            os.makedirs(os.path.dirname(image_path_to), exist_ok=True)
            os.makedirs(os.path.dirname(label_path_to), exist_ok=True)
            os.makedirs(os.path.dirname(mask_path_to), exist_ok=True)
            os.system('cp %s %s' % (image_path_from, image_path_to))
            os.system('cp %s %s' % (label_path_from, label_path_to))
            os.system('cp %s %s' % (mask_path_from, mask_path_to))

        for test_item in tqdm(test_list):
            image_path_from = test_image_folder + test_item + '.png'
            label_path_from = test_label_folder + test_item + '.png'
            mask_path_from = test_mask_folder + test_item + '.png'
            image_path_to = os.path.join(target_folder, tumor_type, 'test/images/', test_item + '.png')
            label_path_to = os.path.join(target_folder, tumor_type, 'test/labels/', test_item + '.png')
            mask_path_to = os.path.join(target_folder, tumor_type, 'test/masks/', test_item + '.png')

            os.makedirs(os.path.dirname(image_path_to), exist_ok=True)
            os.makedirs(os.path.dirname(label_path_to), exist_ok=True)
            os.makedirs(os.path.dirname(mask_path_to), exist_ok=True)
            os.system('cp %s %s' % (image_path_from, image_path_to))
            os.system('cp %s %s' % (label_path_from, label_path_to))
            os.system('cp %s %s' % (mask_path_from, mask_path_to))

    return

def subset_patchify_GLySAC_data_by_tumor(imsize: int):
    train_image_folder = '../../data/GLySAC/train/images/'
    train_label_folder = '../../data/GLySAC/train/labels/'
    test_image_folder = '../../data/GLySAC/test/images/'
    test_label_folder = '../../data/GLySAC/test/labels/'

    target_folder = '../../data/GLySAC/GLySACByTumor_%sx%s/' % (imsize, imsize)

    for tumor_type in ['Tumor', 'Normal']:
        train_list = GLySAC_Organ2FileID[tumor_type]['train']
        test_list = GLySAC_Organ2FileID[tumor_type]['test']

        for train_item in tqdm(train_list):
            image_path_from = train_image_folder + train_item + '.png'
            label_path_from = train_label_folder + train_item + '.png'

            image = cv2.cvtColor(cv2.imread(image_path_from, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
            label = cv2.imread(label_path_from, cv2.IMREAD_UNCHANGED)
            image_h, image_w = image.shape[:2]

            for h_chunk in range(image_h // imsize):
                for w_chunk in range(image_w // imsize):
                    h = h_chunk * imsize
                    w = w_chunk * imsize

                    image_path_to = os.path.join(target_folder, tumor_type, 'train/images/', train_item + '_H%sW%s.png' % (h, w))
                    label_path_to = os.path.join(target_folder, tumor_type, 'train/labels/', train_item + '_H%sW%s.png' % (h, w))
                    mask_path_to = os.path.join(target_folder, tumor_type, 'train/masks/', train_item + '_H%sW%s.png' % (h, w))

                    os.makedirs(os.path.dirname(image_path_to), exist_ok=True)
                    os.makedirs(os.path.dirname(label_path_to), exist_ok=True)
                    os.makedirs(os.path.dirname(mask_path_to), exist_ok=True)

                    h_begin = max(h, 0)
                    w_begin = max(w, 0)
                    h_end = min(h + imsize, image_h)
                    w_end = min(w + imsize, image_w)

                    image_patch = image[h_begin:h_end, w_begin:w_end, :]
                    label_patch = label[h_begin:h_end, w_begin:w_end]
                    mask_patch = label_patch > 0

                    cv2.imwrite(image_path_to, cv2.cvtColor(image_patch, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(label_path_to, label_patch)
                    cv2.imwrite(mask_path_to, np.uint8(mask_patch * 255))

        for test_item in tqdm(test_list):
            image_path_from = test_image_folder + test_item + '.png'
            label_path_from = test_label_folder + test_item + '.png'

            image = cv2.cvtColor(cv2.imread(image_path_from, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
            label = cv2.imread(label_path_from, cv2.IMREAD_UNCHANGED)
            image_h, image_w = image.shape[:2]

            for h_chunk in range(image_h // imsize):
                for w_chunk in range(image_w // imsize):
                    h = h_chunk * imsize
                    w = w_chunk * imsize

                    image_path_to = os.path.join(target_folder, tumor_type, 'test/images/', test_item + '_H%sW%s.png' % (h, w))
                    label_path_to = os.path.join(target_folder, tumor_type, 'test/labels/', test_item + '_H%sW%s.png' % (h, w))
                    mask_path_to = os.path.join(target_folder, tumor_type, 'test/masks/', test_item + '_H%sW%s.png' % (h, w))

                    os.makedirs(os.path.dirname(image_path_to), exist_ok=True)
                    os.makedirs(os.path.dirname(label_path_to), exist_ok=True)
                    os.makedirs(os.path.dirname(mask_path_to), exist_ok=True)

                    h_begin = max(h, 0)
                    w_begin = max(w, 0)
                    h_end = min(h + imsize, image_h)
                    w_end = min(w + imsize, image_w)

                    image_patch = image[h_begin:h_end, w_begin:w_end, :]
                    label_patch = label[h_begin:h_end, w_begin:w_end]
                    mask_patch = label_patch > 0

                    cv2.imwrite(image_path_to, cv2.cvtColor(image_patch, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(label_path_to, label_patch)
                    cv2.imwrite(mask_path_to, np.uint8(mask_patch * 255))

    return


if __name__ == '__main__':

    # For comparisons
    process_GLySAC_data()
    subset_GLySAC_data_by_tumor()
    subset_patchify_GLySAC_data_by_tumor(imsize=200)

    # For our pipeline
    patchify_GLySAC_data_by_tumor_cell_centric(patch_size=96)
