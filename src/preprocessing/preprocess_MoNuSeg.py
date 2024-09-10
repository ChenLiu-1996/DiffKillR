'''
Read annotations from xml file, find the label maps, and patchify around them.
images are in .tif format, RGB, 1000x1000.

'''
import cv2
import os
import numpy as np
from typing import Tuple
from tqdm import tqdm
from glob import glob
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
    - label: A binary image mask.
    - centroids: A list of centroids for each polygon/cell.
    """

    label = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    centroids = []
    for idx, cell in enumerate(tqdm(verts_list)):
        # cell is shape (n, 2)

        cell_mask = polygon2mask(label.shape, cell)
        label = np.maximum(label, cell_mask).astype(np.uint8)

        centroid = skimage.measure.centroid(cell_mask)
        #centroid = np.argwhere(cell_mask > 0).sum(0) / (cell_mask > 0).sum()
        centroids.append((int(centroid[0]), int(centroid[1])))

    return label, centroids

def process_MoNuSeg_data():
    '''
    images are in .tif format, RGB, 1000x1000.
    '''

    for subset in ['test', 'train']:

        if subset == 'train':
            image_folder = '../../external_data/MoNuSeg/MoNuSegTrainData/Tissue Images'
            annotation_folder = '../../external_data/MoNuSeg/MoNuSegTrainData/Annotations'

            out_image_folder = '../../data/MoNuSeg/MoNuSegTrainData/images/'
            out_mask_folder = '../../data/MoNuSeg/MoNuSegTrainData/masks/'

        else:
            image_folder = '../../external_data/MoNuSeg/MoNuSegTestData'
            annotation_folder = '../../external_data/MoNuSeg/MoNuSegTestData'

            out_image_folder = '../../data/MoNuSeg/MoNuSegTestData/images/'
            out_mask_folder = '../../data/MoNuSeg/MoNuSegTestData/masks/'

        annotation_files = sorted(glob(f'{annotation_folder}/*.xml'))
        image_files = sorted(glob(f'{image_folder}/*.tif'))

        all_verts_list = []

        for i, annotation_file in enumerate(tqdm(annotation_files)):
            image_id = os.path.basename(annotation_file).split('.')[0]
            image_file = f'{image_folder}/{image_id}.tif'
            if image_file not in image_files:
                print(f'Image file {image_file} not found.')
                continue

            image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            assert len(image.shape) == 3
            assert image.shape[-1] == 3

            # Read the annotation xml.
            verts_list, region_id_list = load_MoNuSeg_annotation(annotation_file)
            print('Done reading annotation for image %s' % image_id)
            print('Number of annotated cells: %d' % len(verts_list))
            all_verts_list.extend(verts_list)

            # Produce label from annotation for this image.
            mask, centroids_list = annotation_to_label(verts_list, image)

            os.makedirs(out_image_folder, exist_ok=True)
            os.makedirs(out_mask_folder, exist_ok=True)

            out_image_path = out_image_folder + '/' + image_id + '.png'
            out_mask_path = out_mask_folder + '/' + image_id + '.png'

            assert np.max(mask) in [0, 1]

            cv2.imwrite(out_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(out_mask_path, np.uint8(mask * 255))

    return

def patchify_MoNuSeg_data_by_cancer_mask_centric(imsize: int = 96, background_ratio: float = 0.5):
    '''
    images are in .tif format, RGB, 1000x1000.
    '''

    for cancer_type in ['Breast', 'Colon', 'Prostate']:
        for subset in ['test', 'train']:

            if subset == 'train':
                image_folder = '../../external_data/MoNuSeg/MoNuSegTrainData/Tissue Images'
                annotation_folder = '../../external_data/MoNuSeg/MoNuSegTrainData/Annotations'

                out_image_folder = f'../../data/MoNuSeg/MoNuSegByCancer_patch_{imsize}x{imsize}/{cancer_type}/train/images/'
                out_mask_folder = f'../../data/MoNuSeg/MoNuSegByCancer_patch_{imsize}x{imsize}/{cancer_type}/train/masks/'
                out_bg_image_folder = f'../../data/MoNuSeg/MoNuSegByCancer_patch_{imsize}x{imsize}/{cancer_type}/train/background_images/'

                image_id_list = MoNuSeg_Organ2FileID[cancer_type]['train']

            else:
                image_folder = '../../external_data/MoNuSeg/MoNuSegTestData/'
                annotation_folder = '../../external_data/MoNuSeg/MoNuSegTestData/'

                out_image_folder = f'../../data/MoNuSeg/MoNuSegByCancer_patch_{imsize}x{imsize}/{cancer_type}/test/images/'
                out_mask_folder = f'../../data/MoNuSeg/MoNuSegByCancer_patch_{imsize}x{imsize}/{cancer_type}/test/masks/'
                out_bg_image_folder = f'../../data/MoNuSeg/MoNuSegByCancer_patch_{imsize}x{imsize}/{cancer_type}/test/background_images/'

                image_id_list = MoNuSeg_Organ2FileID[cancer_type]['test']

            for image_id in tqdm(image_id_list):
                image_file = os.path.join(image_folder, image_id + '.tif')
                annotation_file = os.path.join(annotation_folder, image_id + '.xml')

                image = cv2.cvtColor(cv2.imread(image_file, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
                assert len(image.shape) == 3
                assert image.shape[-1] == 3
                image_h, image_w = image.shape[:2]

                # Read the annotation xml.
                verts_list, region_id_list = load_MoNuSeg_annotation(annotation_file)
                print('Done reading annotation for image %s' % image_id)
                print('Number of annotated cells: %d' % len(verts_list))

                # Produce label from annotation for this image.
                mask, centroids_list = annotation_to_label(verts_list, image)
                num_cells = len(centroids_list)

                # Patchify and save the cell images and masks.
                for coord in centroids_list:
                    h = int(coord[0] - imsize / 2)
                    w = int(coord[1] - imsize / 2)

                    h = min(h, image_h - imsize - 1)
                    h = max(h, 0)
                    w = min(w, image_w - imsize - 1)
                    w = max(w, 0)

                    cell_file_name = f'{image_id}_H{h}W{w}_patch_{imsize}x{imsize}.png'
                    bg_image_path_to = os.path.join(out_image_folder, cell_file_name)
                    mask_path_to = os.path.join(out_mask_folder, cell_file_name)
                    os.makedirs(os.path.dirname(bg_image_path_to), exist_ok=True)
                    os.makedirs(os.path.dirname(mask_path_to), exist_ok=True)

                    image_patch = image[h : h+imsize, w : w+imsize, :]
                    mask_patch = mask[h : h+imsize, w : w+imsize]

                    assert mask_patch.min() in [0, 1] and mask_patch.max() in [0, 1]

                    cv2.imwrite(bg_image_path_to, cv2.cvtColor(image_patch, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(mask_path_to, np.uint8(mask_patch * 255))

                # Find background patches that do not contain cells.
                background_locs = find_background_patch_locs(mask, patch_size=imsize)
                num_backgrounds = int(num_cells * background_ratio)
                if len(background_locs) <= 0:
                    print('[Background] No candidate patches found for image %s' % image_id)
                    continue

                np.random.seed(1)
                background_idx = np.random.choice(len(background_locs), size=num_backgrounds, replace=False)
                selected_background_locs = [background_locs[idx] for idx in background_idx]

                # Save background patches.
                for (h, w) in selected_background_locs:
                    bg_file_name = f'{image_id}_H{h}W{w}_patch_{imsize}x{imsize}.png'
                    bg_image_patch = image[h : h+imsize, w : w+imsize]

                    bg_image_path_to = os.path.join(out_bg_image_folder, bg_file_name)
                    os.makedirs(os.path.dirname(bg_image_path_to), exist_ok=True)
                    cv2.imwrite(bg_image_path_to, cv2.cvtColor(bg_image_patch, cv2.COLOR_RGB2BGR))

    return


def find_background_patch_locs(label, patch_size) -> list[str]:
    '''
    TODO: Find background regions given label/mask of the image.
    Args:
        label: np.array of shape (H, W) for binary mask
        patch_size: size of the patch (e.g. 96)

    Returns:
        candidate_patches: list of tuple (h, w) recording the top left corner of the background patches
    '''
    # pixel by pixel scan
    cnts = 0
    stride = 1
    candidate_patches = []
    for h in range(0, label.shape[0], stride):
        for w in range(0, label.shape[1], stride):
            patch_label = label[h:h+patch_size, w:w+patch_size]
            if patch_label.shape != (patch_size, patch_size):
                continue

            # check if the patch intersects with any label patch
            overlap_threshold = 0
            if np.sum(patch_label) <= overlap_threshold:
                cnts += 1
                candidate_patch = (h, w)
                candidate_patches.append(candidate_patch)

    print('[Background] Number of candidate patches: %d' % len(candidate_patches))

    return candidate_patches


def subset_MoNuSeg_data_by_cancer():
    train_image_folder = '../../data/MoNuSeg/MoNuSegTrainData/images/'
    train_mask_folder = '../../data/MoNuSeg/MoNuSegTrainData/masks/'
    test_image_folder = '../../data/MoNuSeg/MoNuSegTestData/images/'
    test_mask_folder = '../../data/MoNuSeg/MoNuSegTestData/masks/'

    target_folder = '../../data/MoNuSeg/MoNuSegByCancer/'

    for cancer_type in ['Breast', 'Colon', 'Prostate']:

        train_list = MoNuSeg_Organ2FileID[cancer_type]['train']
        test_list = MoNuSeg_Organ2FileID[cancer_type]['test']

        for train_item in tqdm(train_list):
            image_path_from = os.path.join(train_image_folder, train_item + '.png')
            mask_path_from = os.path.join(train_mask_folder, train_item + '.png')
            image_path_to = os.path.join(target_folder, cancer_type, 'train', 'images', train_item + '.png')
            mask_path_to = os.path.join(target_folder, cancer_type, 'train', 'masks', train_item + '.png')

            os.makedirs(os.path.dirname(image_path_to), exist_ok=True)
            os.makedirs(os.path.dirname(mask_path_to), exist_ok=True)
            os.system('cp %s %s' % (image_path_from, image_path_to))
            os.system('cp %s %s' % (mask_path_from, mask_path_to))

        for test_item in tqdm(test_list):
            image_path_from = os.path.join(test_image_folder, test_item + '.png')
            mask_path_from = os.path.join(test_mask_folder, test_item + '.png')
            image_path_to = os.path.join(target_folder, cancer_type, 'test', 'images', test_item + '.png')
            mask_path_to = os.path.join(target_folder, cancer_type, 'test', 'masks', test_item + '.png')

            os.makedirs(os.path.dirname(image_path_to), exist_ok=True)
            os.makedirs(os.path.dirname(mask_path_to), exist_ok=True)
            os.system('cp %s %s' % (image_path_from, image_path_to))
            os.system('cp %s %s' % (mask_path_from, mask_path_to))

    return

def subset_patchify_MoNuSeg_data_by_cancer(imsize: int):
    train_image_folder = '../../data/MoNuSeg/MoNuSegTrainData/images/'
    train_mask_folder = '../../data/MoNuSeg/MoNuSegTrainData/masks/'
    test_image_folder = '../../data/MoNuSeg/MoNuSegTestData/images/'
    test_mask_folder = '../../data/MoNuSeg/MoNuSegTestData/masks/'

    target_folder = '../../data/MoNuSeg/MoNuSegByCancer_%sx%s/' % (imsize, imsize)

    for cancer_type in ['Breast', 'Colon', 'Prostate']:
        train_list = MoNuSeg_Organ2FileID[cancer_type]['train']
        test_list = MoNuSeg_Organ2FileID[cancer_type]['test']

        for train_item in tqdm(train_list):
            image_path_from = os.path.join(train_image_folder, train_item + '.png')
            mask_path_from = os.path.join(train_mask_folder, train_item + '.png')

            image = cv2.imread(image_path_from)
            mask = cv2.imread(mask_path_from)
            image_h, image_w = image.shape[:2]

            for h_chunk in range(image_h // imsize):
                for w_chunk in range(image_w // imsize):
                    h = h_chunk * imsize
                    w = w_chunk * imsize

                    h = min(h, image_h - imsize - 1)
                    h = max(h, 0)
                    w = min(w, image_w - imsize - 1)
                    w = max(w, 0)

                    image_path_to = os.path.join(target_folder, cancer_type, 'train', 'images', train_item + f'_H{h}W{w}.png')
                    mask_path_to = os.path.join(target_folder, cancer_type, 'train', 'masks', train_item + f'_H{h}W{w}.png')
                    os.makedirs(os.path.dirname(image_path_to), exist_ok=True)
                    os.makedirs(os.path.dirname(mask_path_to), exist_ok=True)

                    image_patch = image[h : h+imsize, w : w+imsize, :]
                    mask_patch = mask[h : h+imsize, w : w+imsize]

                    cv2.imwrite(image_path_to, image_patch)
                    cv2.imwrite(mask_path_to, mask_patch)

        for test_item in tqdm(test_list):
            image_path_from = os.path.join(test_image_folder, test_item + '.png')
            mask_path_from = os.path.join(test_mask_folder, test_item + '.png')

            image = cv2.imread(image_path_from)
            mask = cv2.imread(mask_path_from)
            image_h, image_w = image.shape[:2]

            for h_chunk in range(image_h // imsize):
                for w_chunk in range(image_w // imsize):
                    h = h_chunk * imsize
                    w = w_chunk * imsize

                    h = min(h, image_h - imsize - 1)
                    h = max(h, 0)
                    w = min(w, image_w - imsize - 1)
                    w = max(w, 0)

                    image_path_to = os.path.join(target_folder, cancer_type, 'test', 'images', test_item + f'_H{h}W{w}.png')
                    mask_path_to = os.path.join(target_folder, cancer_type, 'test', 'masks', test_item + f'_H{h}W{w}.png')
                    os.makedirs(os.path.dirname(image_path_to), exist_ok=True)
                    os.makedirs(os.path.dirname(mask_path_to), exist_ok=True)

                    image_patch = image[h : h+imsize, w : w+imsize, :]
                    mask_patch = mask[h : h+imsize, w : w+imsize]

                    cv2.imwrite(image_path_to, image_patch)
                    cv2.imwrite(mask_path_to, mask_patch)
    return

def subset_patchify_MoNuSeg_data_by_cancer_intraimage(imsize: int):
    test_image_folder = '../../data/MoNuSeg/MoNuSegTestData/images/'
    test_mask_folder = '../../data/MoNuSeg/MoNuSegTestData/masks/'

    for cancer_type in ['Breast', 'Colon', 'Prostate']:
        test_list = MoNuSeg_Organ2FileID[cancer_type]['test']

        for percentage in [5, 20, 50]:
            target_folder = '../../data/MoNuSeg/MoNuSegByCancer_intraimage%dpct_%sx%s/' % (percentage, imsize, imsize)

            for test_item_count, test_item in enumerate(tqdm(test_list)):
                image_path_from = os.path.join(test_image_folder, test_item + '.png')
                mask_path_from = os.path.join(test_mask_folder, test_item + '.png')

                image = cv2.imread(image_path_from)
                mask = cv2.imread(mask_path_from)
                image_h, image_w = image.shape[:2]

                total_count = (image_h // imsize) * (image_w // imsize)
                target_count = int(np.ceil(percentage * total_count / 100))
                curr_count = 0

                # Also track the "effective" image/mask pair for evaluation.
                image_effective = np.zeros_like(image)
                mask_effective = np.zeros_like(mask)

                for h_chunk in range(image_h // imsize):
                    for w_chunk in range(image_w // imsize):
                        h = h_chunk * imsize
                        w = w_chunk * imsize

                        h = min(h, image_h - imsize - 1)
                        h = max(h, 0)
                        w = min(w, image_w - imsize - 1)
                        w = max(w, 0)

                        image_patch = image[h : h+imsize, w : w+imsize, :]
                        mask_patch = mask[h : h+imsize, w : w+imsize]

                        if curr_count < target_count:
                            # 1. Save the image/mask pair to the train folder.
                            image_path_to = os.path.join(target_folder, cancer_type, f'img{test_item_count}_train', 'images', test_item + f'_H{h}W{w}.png')
                            mask_path_to = os.path.join(target_folder, cancer_type, f'img{test_item_count}_train', 'masks', test_item + f'_H{h}W{w}.png')
                            os.makedirs(os.path.dirname(image_path_to), exist_ok=True)
                            os.makedirs(os.path.dirname(mask_path_to), exist_ok=True)

                            cv2.imwrite(image_path_to, image_patch)
                            cv2.imwrite(mask_path_to, mask_patch)

                            # 2. Save an empty image/mask pair to the test folder.
                            empty_image_patch = image_patch * 0
                            empty_mask_patch = mask_patch * 0
                            empty_image_path_to = os.path.join(target_folder, cancer_type, f'img{test_item_count}_test', 'images', test_item + f'_H{h}W{w}.png')
                            empty_mask_path_to = os.path.join(target_folder, cancer_type, f'img{test_item_count}_test', 'masks', test_item + f'_H{h}W{w}.png')
                            os.makedirs(os.path.dirname(empty_image_path_to), exist_ok=True)
                            os.makedirs(os.path.dirname(empty_mask_path_to), exist_ok=True)

                            cv2.imwrite(empty_image_path_to, empty_image_patch)
                            cv2.imwrite(empty_mask_path_to, empty_mask_patch)

                        else:
                            # Save the image/mask pair to the test folder.
                            image_path_to = os.path.join(target_folder, cancer_type, f'img{test_item_count}_test', 'images', test_item + f'_H{h}W{w}.png')
                            mask_path_to = os.path.join(target_folder, cancer_type, f'img{test_item_count}_test', 'masks', test_item + f'_H{h}W{w}.png')
                            os.makedirs(os.path.dirname(image_path_to), exist_ok=True)
                            os.makedirs(os.path.dirname(mask_path_to), exist_ok=True)

                            cv2.imwrite(image_path_to, image_patch)
                            cv2.imwrite(mask_path_to, mask_patch)

                            # Update the "effective" image/mask pair.
                            image_effective[h : h+imsize, w : w+imsize, :] = image[h : h+imsize, w : w+imsize, :]
                            mask_effective[h : h+imsize, w : w+imsize] = mask[h : h+imsize, w : w+imsize]

                        curr_count += 1

                # Save the "effective" image/mask pair.
                image_effective_path_to = os.path.join(target_folder, cancer_type, f'img{test_item_count}_test', test_item + '_effective_image.png')
                mask_effective_path_to = os.path.join(target_folder, cancer_type, f'img{test_item_count}_test', test_item + '_effective_mask.png')
                cv2.imwrite(image_effective_path_to, image_effective)
                cv2.imwrite(mask_effective_path_to, mask_effective)

    return


if __name__ == '__main__':
    # For our pipeline
    patchify_MoNuSeg_data_by_cancer_mask_centric(imsize=96)

    # For comparison
    process_MoNuSeg_data()
    subset_MoNuSeg_data_by_cancer()
    subset_patchify_MoNuSeg_data_by_cancer(imsize=200)
    subset_patchify_MoNuSeg_data_by_cancer_intraimage(imsize=200)
