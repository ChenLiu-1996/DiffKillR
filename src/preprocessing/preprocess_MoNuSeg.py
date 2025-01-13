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
from typing import Tuple
from tqdm import tqdm
from glob import glob
from matplotlib import pyplot as plt
from skimage.draw import polygon2mask
import skimage.measure
from Metas import MoNuSeg_Organ2FileID


def load_MoNuSeg_annotation(xml_path: str) -> list[np.ndarray]:
    '''
        Return a list of vertices for each polygon in the xml file
        Each polygon is np.array of shape (n, 2), where n is the number of vertices

        No classes in this dataset, so we will just return the vertices.
    '''
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    root = tree.getroot()
    regions_root = root.find('.//Regions')
    regions = regions_root.findall('.//Region')

    verts_list = []
    region_id_list = []
    cnt_invalid = 0

    for region in regions:
        # Skip polygons with area less than 1.0.
        region_area = float(region.attrib['Area'])
        if region_area < 1.0:
            cnt_invalid += 1
            continue

        vertices = region.findall('.//Vertex')
        # Skip polygons with less than 3 vertices.
        if len(vertices) < 3:
            cnt_invalid += 1
            continue

        region_id = region.attrib['Id']
        region_id_list.append(int(region_id))
        verts = []
        for vertex in vertices:
            x = float(vertex.attrib['X']) # TODO: maybe round to int?
            y = float(vertex.attrib['Y'])
            verts.append([y, x])
        verts = np.array(verts) # shape (n, 2)
        verts_list.append(verts)

    print('Total polygons: %d, Invalid polygons: %d' % (len(regions), cnt_invalid))

    return (verts_list, region_id_list)


def annotation_to_label(verts_list: list[np.ndarray],
                        image: np.array) -> Tuple[np.array, dict]:
    """
    Converts polygon annotations to a labeled image and calculates centroids of the polygons.

    Parameters:
    - verts_list: A list of vertices for each polygon/cell.
    - image: The image for which annotations are being converted.

    Returns:
    - label: An instance label. Different cells have different values.
    - centroids: A list of centroids for each polygon/cell.
    """

    label = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint16)  # uint16 is essential!
    centroids = []
    for i, cell in enumerate(tqdm(verts_list)):
        # cell is shape (n, 2)

        cell_label = polygon2mask(label.shape, cell) * (i + 1)
        cell_label = cell_label.astype(np.uint16)  # uint16 is essential!
        label = np.maximum(label, cell_label)

        centroid = skimage.measure.centroid(cell_label)
        centroids.append((int(centroid[0]), int(centroid[1])))

    assert len(centroids) < 2**16
    return label, centroids

def process_MoNuSeg_data():
    '''
    images are in .tif format, RGB, 1000x1000.
    '''

    for subset in ['train', 'test']:

        if subset == 'train':
            image_folder = '../../external_data/MoNuSeg/MoNuSegTrainData/Tissue Images'
            annotation_folder = '../../external_data/MoNuSeg/MoNuSegTrainData/Annotations'

        else:
            image_folder = '../../external_data/MoNuSeg/MoNuSegTestData'
            annotation_folder = '../../external_data/MoNuSeg/MoNuSegTestData'

        out_image_folder = f'../../data/MoNuSeg/{subset}/images/'
        out_label_folder = f'../../data/MoNuSeg/{subset}/labels/'
        out_mask_folder = f'../../data/MoNuSeg/{subset}/masks/'
        out_stats_folder = f'../../data/MoNuSeg/{subset}/stats/'

        annotation_files = sorted(glob(f'{annotation_folder}/*.xml'))
        image_files = sorted(glob(f'{image_folder}/*.tif'))
        all_verts_list = []

        for annotation_file in tqdm(annotation_files):
            image_id = os.path.basename(annotation_file).split('.')[0]
            image_file = f'{image_folder}/{image_id}.tif'
            if image_file not in image_files:
                print(f'Image file {image_file} not found.')
                continue

            image = cv2.cvtColor(cv2.imread(image_file, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
            assert len(image.shape) == 3
            assert image.shape[-1] == 3

            # Read the annotation xml.
            verts_list, _ = load_MoNuSeg_annotation(annotation_file)
            print('Done reading annotation for image %s' % image_id)
            print('Number of annotated cells: %d' % len(verts_list))
            all_verts_list.extend(verts_list)

            # Produce label from annotation for this image.
            label, _ = annotation_to_label(verts_list, image)

            os.makedirs(out_image_folder, exist_ok=True)
            os.makedirs(out_label_folder, exist_ok=True)
            os.makedirs(out_mask_folder, exist_ok=True)

            out_image_path = os.path.join(out_image_folder, image_id + '.png')
            out_label_path = os.path.join(out_label_folder, image_id + '.png')
            out_mask_path = os.path.join(out_mask_folder, image_id + '.png')

            assert label.dtype == 'uint16'
            mask = label > 0

            cv2.imwrite(out_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(out_label_path, label)
            cv2.imwrite(out_mask_path, np.uint8(mask * 255))

        # Size statistics about the annotated cells.
        h_list, w_list = [], []
        for cell_verts in all_verts_list:
            h_min = np.min(cell_verts[:, 0])
            h_max = np.max(cell_verts[:, 0])
            w_min = np.min(cell_verts[:, 1])
            w_max = np.max(cell_verts[:, 1])

            h_list.append(h_max - h_min + 1)
            w_list.append(w_max - w_min + 1)

        print('MoNuSeg statistics (%s set)' % subset)
        print('Mean height: %d, Mean width: %d' % (np.mean(h_list), np.mean(w_list)))
        print('Std height: %d, Std width: %d' % (np.std(h_list), np.std(w_list)))

        os.makedirs(out_stats_folder, exist_ok=True)
        stats_fig_save_path = os.path.join(out_stats_folder, f'cell_size_distribution_{subset}.png')
        stats_arr_save_path = os.path.join(out_stats_folder, f'cell_sizes_{subset}.npz')
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

def subset_MoNuSeg_data_by_cancer():
    train_image_folder = '../../data/MoNuSeg/train/images/'
    train_label_folder = '../../data/MoNuSeg/train/labels/'
    train_mask_folder = '../../data/MoNuSeg/train/masks/'
    test_image_folder = '../../data/MoNuSeg/test/images/'
    test_label_folder = '../../data/MoNuSeg/test/labels/'
    test_mask_folder = '../../data/MoNuSeg/test/masks/'

    target_folder = '../../data/MoNuSeg/MoNuSegByCancer/'

    for cancer_type in MoNuSeg_Organ2FileID.keys():
        for subset in ['train', 'test']:
            train_list = MoNuSeg_Organ2FileID[cancer_type]['train']
            test_list = MoNuSeg_Organ2FileID[cancer_type]['test']

            # NOTE: When we partition by cancer,
            # we split the samples for organs that were used exclusively for train or test.
            # Therefore, the original "train" or "test" split no longer hold true.
            for item in tqdm(train_list + test_list):
                try:
                    image_path_from = os.path.join(train_image_folder, item + '.png')
                    label_path_from = os.path.join(train_label_folder, item + '.png')
                    mask_path_from = os.path.join(train_mask_folder, item + '.png')
                    assert os.path.isfile(image_path_from) and os.path.isfile(label_path_from) and os.path.isfile(mask_path_from)
                except:
                    image_path_from = os.path.join(test_image_folder, item + '.png')
                    label_path_from = os.path.join(test_label_folder, item + '.png')
                    mask_path_from = os.path.join(test_mask_folder, item + '.png')
                    assert os.path.isfile(image_path_from) and os.path.isfile(label_path_from) and os.path.isfile(mask_path_from)

                image_path_to = os.path.join(target_folder, cancer_type, subset, 'images', item + '.png')
                label_path_to = os.path.join(target_folder, cancer_type, subset, 'labels', item + '.png')
                mask_path_to = os.path.join(target_folder, cancer_type, subset, 'masks', item + '.png')

                os.makedirs(os.path.dirname(image_path_to), exist_ok=True)
                os.makedirs(os.path.dirname(label_path_to), exist_ok=True)
                os.makedirs(os.path.dirname(mask_path_to), exist_ok=True)
                os.system('cp %s %s' % (image_path_from, image_path_to))
                os.system('cp %s %s' % (label_path_from, label_path_to))
                os.system('cp %s %s' % (mask_path_from, mask_path_to))

    return

def subset_patchify_MoNuSeg_data_by_cancer(imsize: int):
    train_image_folder = '../../data/MoNuSeg/train/images/'
    train_label_folder = '../../data/MoNuSeg/train/labels/'
    test_image_folder = '../../data/MoNuSeg/test/images/'
    test_label_folder = '../../data/MoNuSeg/test/labels/'

    target_folder = '../../data/MoNuSeg/MoNuSegByCancer_%sx%s/' % (imsize, imsize)

    for cancer_type in MoNuSeg_Organ2FileID.keys():
        for subset in ['train', 'test']:
            train_list = MoNuSeg_Organ2FileID[cancer_type]['train']
            test_list = MoNuSeg_Organ2FileID[cancer_type]['test']

            # NOTE: When we partition by cancer,
            # we split the samples for organs that were used exclusively for train or test.
            # Therefore, the original "train" or "test" split no longer hold true.
            for item in tqdm(train_list + test_list):
                try:
                    image_path_from = os.path.join(train_image_folder, item + '.png')
                    label_path_from = os.path.join(train_label_folder, item + '.png')
                    assert os.path.isfile(image_path_from) and os.path.isfile(label_path_from)
                except:
                    image_path_from = os.path.join(test_image_folder, item + '.png')
                    label_path_from = os.path.join(test_label_folder, item + '.png')
                    assert os.path.isfile(image_path_from) and os.path.isfile(label_path_from)

                image = cv2.cvtColor(cv2.imread(image_path_from, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
                label = cv2.imread(label_path_from, cv2.IMREAD_UNCHANGED)
                assert label.dtype == 'uint16'

                image_h, image_w = image.shape[:2]

                for h_chunk in range(image_h // imsize):
                    for w_chunk in range(image_w // imsize):
                        h = h_chunk * imsize
                        w = w_chunk * imsize

                        h = min(h, image_h - imsize)
                        h = max(h, 0)
                        w = min(w, image_w - imsize)
                        w = max(w, 0)

                        image_path_to = os.path.join(target_folder, cancer_type, subset, 'images', item + f'_H{h}W{w}.png')
                        label_path_to = os.path.join(target_folder, cancer_type, subset, 'labels', item + f'_H{h}W{w}.png')
                        mask_path_to = os.path.join(target_folder, cancer_type, subset, 'masks', item + f'_H{h}W{w}.png')

                        os.makedirs(os.path.dirname(image_path_to), exist_ok=True)
                        os.makedirs(os.path.dirname(label_path_to), exist_ok=True)
                        os.makedirs(os.path.dirname(mask_path_to), exist_ok=True)

                        image_patch = image[h : h+imsize, w : w+imsize, :]
                        label_patch = label[h : h+imsize, w : w+imsize]
                        mask_patch = label_patch > 0
                        assert label_patch.dtype == 'uint16'

                        cv2.imwrite(image_path_to, cv2.cvtColor(image_patch, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(label_path_to, label_patch)
                        cv2.imwrite(mask_path_to, np.uint8(mask_patch * 255))

    return

def patchify_MoNuSeg_data_by_cancer_cell_centric(patch_size: int, background_ratio: float = 0.5):
    '''
    images are in .tif format, RGB, 1000x1000.
    '''

    for cancer_type in MoNuSeg_Organ2FileID.keys():
        for subset in ['train', 'test']:

            # NOTE: When we partition by cancer,
            # we split the samples for organs that were used exclusively for train or test.
            # Therefore, the original "train" or "test" split no longer hold true.
            train_image_folder = '../../external_data/MoNuSeg/MoNuSegTrainData/Tissue Images'
            train_annotation_folder = '../../external_data/MoNuSeg/MoNuSegTrainData/Annotations'
            test_image_folder = '../../external_data/MoNuSeg/MoNuSegTestData/'
            test_annotation_folder = '../../external_data/MoNuSeg/MoNuSegTestData/'

            out_image_folder = f'../../data/MoNuSeg/MoNuSegByCancer_patch_{patch_size}x{patch_size}/{cancer_type}/{subset}/images/'
            out_label_folder = f'../../data/MoNuSeg/MoNuSegByCancer_patch_{patch_size}x{patch_size}/{cancer_type}/{subset}/labels/'
            out_mask_folder = f'../../data/MoNuSeg/MoNuSegByCancer_patch_{patch_size}x{patch_size}/{cancer_type}/{subset}/masks/'
            out_bg_image_folder = f'../../data/MoNuSeg/MoNuSegByCancer_patch_{patch_size}x{patch_size}/{cancer_type}/{subset}/background_images/'
            out_stats_folder = f'../../data/MoNuSeg/MoNuSegByCancer/{cancer_type}/{subset}/stats/'

            image_id_list = MoNuSeg_Organ2FileID[cancer_type][subset]
            all_verts_list = []

            for image_id in tqdm(image_id_list):
                try:
                    image_file = os.path.join(train_image_folder, image_id + '.tif')
                    annotation_file = os.path.join(train_annotation_folder, image_id + '.xml')
                    assert os.path.isfile(image_file) and os.path.isfile(annotation_file)
                except:
                    image_file = os.path.join(test_image_folder, image_id + '.tif')
                    annotation_file = os.path.join(test_annotation_folder, image_id + '.xml')
                    assert os.path.isfile(image_file) and os.path.isfile(annotation_file)

                image = cv2.cvtColor(cv2.imread(image_file, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
                assert len(image.shape) == 3
                assert image.shape[-1] == 3
                image_h, image_w = image.shape[:2]

                # Read the annotation xml.
                verts_list, region_id_list = load_MoNuSeg_annotation(annotation_file)
                print('Done reading annotation for image %s' % image_id)
                print('Number of annotated cells: %d' % len(verts_list))
                all_verts_list.extend(verts_list)

                # Produce label from annotation for this image.
                label, centroids_list = annotation_to_label(verts_list, image)
                num_cells = len(centroids_list)

                # Patchify and save the cell images and labels.
                for coord in centroids_list:
                    h = int(coord[0] - patch_size // 2)
                    w = int(coord[1] - patch_size // 2)

                    h = min(h, image_h - patch_size)
                    h = max(h, 0)
                    w = min(w, image_w - patch_size)
                    w = max(w, 0)

                    os.makedirs(out_image_folder, exist_ok=True)
                    os.makedirs(out_label_folder, exist_ok=True)
                    os.makedirs(out_mask_folder, exist_ok=True)

                    cell_file_name = f'{image_id}_H{h}W{w}_patch_{patch_size}x{patch_size}.png'
                    bg_image_path_to = os.path.join(out_image_folder, cell_file_name)
                    label_path_to = os.path.join(out_label_folder, cell_file_name)
                    mask_path_to = os.path.join(out_mask_folder, cell_file_name)

                    image_patch = image[h : h+patch_size, w : w+patch_size, :]
                    label_patch = label[h : h+patch_size, w : w+patch_size]
                    mask_patch = label_patch > 0
                    assert label_patch.dtype == 'uint16'

                    cv2.imwrite(bg_image_path_to, cv2.cvtColor(image_patch, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(label_path_to, label_patch)
                    cv2.imwrite(mask_path_to, np.uint8(mask_patch * 255))

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

            # Size statistics about the annotated cells.
            h_list, w_list = [], []
            for cell_verts in all_verts_list:
                h_min = np.min(cell_verts[:, 0])
                h_max = np.max(cell_verts[:, 0])
                w_min = np.min(cell_verts[:, 1])
                w_max = np.max(cell_verts[:, 1])

                h_list.append(h_max - h_min + 1)
                w_list.append(w_max - w_min + 1)

            print('MoNuSeg statistics (%s set, %s)' % (subset, cancer_type))
            print('Mean height: %d, Mean width: %d' % (np.mean(h_list), np.mean(w_list)))
            print('Std height: %d, Std width: %d' % (np.std(h_list), np.std(w_list)))

            os.makedirs(out_stats_folder, exist_ok=True)
            stats_fig_save_path = os.path.join(out_stats_folder, f'cell_size_distribution_{subset}.png')
            stats_arr_save_path = os.path.join(out_stats_folder, f'cell_sizes_{subset}.npz')
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


def test():
    aaa = '../../data/MoNuSeg/train/labels/'
    path = glob(aaa + '*.png')[0]

    label = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    # test()

    # For comparisons
    process_MoNuSeg_data()
    subset_MoNuSeg_data_by_cancer()
    subset_patchify_MoNuSeg_data_by_cancer(imsize=200)

    # For our pipeline
    patchify_MoNuSeg_data_by_cancer_cell_centric(patch_size=96)
