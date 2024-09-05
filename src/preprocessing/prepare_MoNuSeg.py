'''
Read annotations from xml file, find the label maps, and patchify around them.
images are in .tif format, RGB, 1000x1000.

Find background patches and save them as well.

ASSUMPTIONS:
executed from the '/src' folder of the project.

'''
import cv2
import os
import numpy as np
from typing import Tuple
from tqdm import tqdm
from glob import glob
from skimage.draw import polygon2mask
import skimage.measure
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt

from Metas import MoNuSeg_Organ2FileID

ROOT_DIR = '/'.join(os.path.realpath(__file__).split('/')[:-3])

def load_MoNuSeg_annotation(xml_path: str) -> Tuple[list[np.ndarray], list[int]]:
    '''
        Return a list of vertices for each polygon in the xml file
        Each polygon is np.array of shape (n, 2), where n is the number of vertices

        No classes in this dataset, so we will just return the vertices.
        Args:
            xml_path: path to the xml file
        Returns:
            verts_list: list of np.arrays, each np.array is a polygon
            region_id_list: list of int, each int is the region id
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
            #verts.append([x, y])
            verts.append([y, x]) # FIXME!: check if this is correct. seems to have fixed the issue.
        verts = np.array(verts) # shape (n, 2)
        verts_list.append(verts)

    print('Total polygons: %d, Invalid polygons: %d' % (len(regions), cnt_invalid))

    return (verts_list, region_id_list)


def annotation_to_label(verts_list: list[np.ndarray],
                        image: np.array,
                        image_id: str,
                        region_id_list: list[int]) -> Tuple[np.array, dict]:
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
        centroids.append((int(centroid[0]), int(centroid[1])))

    return label, centroids


def patchify_and_save(image, image_id, label, centroid_list,
                      patches_folder, patch_size) -> list[str]:
    '''
    Divide the image and label into patches and save them.

    image: original image.
    image_id: id of the image. This should be unique.
    label: binary image mask.
    centroid_list: list of centroids for each polygon/cell.

    Returns:
        file_name_list: list of file names for the patches.

    '''
    file_name_list = []
    for centroid in tqdm(centroid_list, desc='patchifying...'):
        file_name = '%s_H%d_W%d_patch_%dx%d' % (image_id, centroid[0] - patch_size // 2,
                                                centroid[1] - patch_size // 2, patch_size, patch_size)
        file_name_list.append(file_name)
        patch_image_path = '%s/image/%s.png' % (patches_folder, file_name)
        patch_label_path = '%s/label/%s.png' % (patches_folder, file_name)
        patch_colored_label_path = '%s/colored_label/%s.png' % (patches_folder, file_name)
        os.makedirs(os.path.dirname(patch_image_path), exist_ok=True)
        os.makedirs(os.path.dirname(patch_label_path), exist_ok=True)
        os.makedirs(os.path.dirname(patch_colored_label_path), exist_ok=True)


        h_begin = max(centroid[0] - patch_size // 2, 0)
        w_begin = max(centroid[1] - patch_size // 2, 0)
        h_end = min(h_begin + patch_size, image.shape[0])
        w_end = min(w_begin + patch_size, image.shape[1])

        # print('centroid', centroid)
        # print('h, w', h_begin, h_end, w_begin, w_end)

        patch_image = image[h_begin:h_end, w_begin:w_end, :]
        patch_label = label[h_begin:h_end, w_begin:w_end]

        # Handle edge cases: literally on the edge.
        if patch_image.shape != (patch_size, patch_size, 3):
            h_diff = patch_size - patch_image.shape[0]
            w_diff = patch_size - patch_image.shape[1]
            patch_image = np.pad(patch_image,
                                    pad_width=((0, h_diff), (0, w_diff), (0,
                                                                        0)),
                                    mode='constant')
            patch_label = np.pad(patch_label,
                                    pad_width=((0, h_diff), (0, w_diff)),
                                    mode='constant')

        patch_image = cv2.cvtColor(patch_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(patch_image_path, patch_image)
        cv2.imwrite(patch_label_path, patch_label)

        # NOTE: Colored label only used for visual inspection!
        patch_label_colored = np.zeros_like(patch_image)
        patch_label_colored[patch_label == 1] = (0, 0, 255)
        cv2.imwrite(patch_colored_label_path, patch_label_colored)

    return file_name_list


def find_background_patches(label, image, image_id, patches_folder, patch_size) -> list[str]:
    '''
    TODO: Find background given label/mask of the image
    Args:
    label: np.array of shape (H, W) for binary mask
    image: np.array of shape (H, W, 3) for the original image
    image_id: id of the image (e.g. TCGA-HE-7128-01Z-00-DX1)
    patch_folder: save folder for the patches (e.g. ../data/MoNuSeg2018TrainData_patch_96x96/)
    patch_size: size of the patch (e.g. 96)

    Returns:
        candidate_patches: list of tuple (ch, cw) for the background patches
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

            ch = h + patch_size // 2
            cw = w + patch_size // 2

            # check if the patch intersects with any label patch
            # !TODO: might allow some overlap
            overlap_threshold = 0
            if np.sum(patch_label) <= overlap_threshold:
                cnts += 1
                candidate_patch = (ch, cw)
                candidate_patches.append(candidate_patch)
                # patch_label_path = '%s/label/%s_H%d_W%d_background.png' % (
                #     patches_folder, image_id, ch, cw)
                # os.makedirs(os.path.dirname(patch_label_path), exist_ok=True)
                # cv2.imwrite(patch_label_path, patch_label)
                
                # patch_image = image[h:h+patch_size, w:w+patch_size]
                # patch_image_path = '%s/image/%s_H%d_W%d_background.png' % (
                #     patches_folder, image_id, ch, cw)
                # os.makedirs(os.path.dirname(patch_image_path), exist_ok=True)
                # patch_image = cv2.cvtColor(patch_image, cv2.COLOR_RGB2BGR)
                # cv2.imwrite(patch_image_path, patch_image)
    
    print('[Background] Number of candidate patches: %d' % len(candidate_patches))
    
    return candidate_patches
    

def process_MoNuSeg_Traindata(patch_size=96, organ='Breast', background_ratio=0.2):
    '''
        images are in .tif format, RGB, 1000x1000.
    '''
    image_folder = '../../external_data/MoNuSeg/MoNuSegTrainData/Tissue Images'
    annotation_folder = '../../external_data/MoNuSeg/MoNuSegTrainData/Annotations'

    annotation_files = sorted(glob(f'{annotation_folder}/*.xml'))
    image_files = sorted(glob(f'{image_folder}/*.tif'))

    all_verts_list = []
    all_background_files = []
    print('Number of images: %d' % len(annotation_files))

    # df for storing image id(file name + ch, cw), and the corresponding annotation info.
    df = pd.DataFrame(columns=['patch_id', 'type'])

    for i, annotation_file in enumerate(tqdm(annotation_files)):
        image_id = os.path.basename(annotation_file).split('.')[0]
        if image_id not in MoNuSeg_Organ2FileID[organ]['train']:
            continue
        image_file = f'{image_folder}/{image_id}.tif'
        if image_file not in image_files:
            print(f'Image file {image_file} not found.')
            continue

        image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        assert len(image.shape) == 3
        assert image.shape[-1] == 3

        # Read the annotation xml. Vertices is a list of np.arrays, each np.array is a polygon.
        verts_list, region_id_list = load_MoNuSeg_annotation(annotation_file)
        print('Done reading annotation for image %s' % image_id)
        print('Number of annotated cells: %d' % len(verts_list))
        all_verts_list.extend(verts_list)

        # Produce label from annotation for this image.
        label, centroids_list = annotation_to_label(verts_list, image, image_id, region_id_list)

        # Divide the image and label into patches.
        patches_folder = f'{ROOT_DIR}/data/MoNuSeg2018TrainData_patch_%dx%d_%s/' % (patch_size, patch_size, organ)
        os.makedirs(patches_folder, exist_ok=True)
        saved_file_names = patchify_and_save(image, image_id, label, centroids_list, patches_folder, patch_size)

        df = pd.concat([df, pd.DataFrame({'patch_id': saved_file_names, 'type': 'cell'})])

        cell_count = len(all_verts_list)
        # Find background patches and save them.
        # background_patches_folder = '../data/MoNuSeg2018Background_patch_%dx%d/' % (aug_patch_size, 
        #                                                                             aug_patch_size)
        # find_background_and_save(label, image, image_id, 
        #                          background_patches_folder, aug_patch_size)

        candidate_patches = find_background_patches(label, image, image_id, patches_folder, patch_size)
        if len(candidate_patches) <= 0:
            print('[Background] No candidate patches found for image %s' % image_id)
            continue
        sample_size = np.min([int(cell_count*background_ratio), len(candidate_patches)])
        candidate_idx = np.random.choice(len(candidate_patches), size=sample_size, replace=False)
        background_patches = [candidate_patches[idx] for idx in candidate_idx]
        bg_file_names = []
        # Save background patches.
        for (ch, cw) in background_patches:
            bg_file_name = '%s_H%d_W%d_patch_%dx%d' % (image_id, ch, cw, patch_size, patch_size)
            bg_file_names.append(bg_file_name)
            h = ch - patch_size // 2
            w = cw - patch_size // 2
            bg_patch = image[h:h+patch_size, w:w+patch_size]
            bg_patch = cv2.cvtColor(bg_patch, cv2.COLOR_RGB2BGR)
            save_path = '%s/image/%s.png' % (patches_folder, bg_file_name)
            # print(f'[Background] Saving background patch to {save_path}')
            cv2.imwrite(save_path, bg_patch)
        all_background_files.extend(bg_file_names)
        # Save background patches annotations to df.
        df = pd.concat([df, pd.DataFrame({'patch_id': bg_file_names, 'type': 'background'})])
    df.to_csv(f'{patches_folder}/class_labels.csv', index=False)

    print('Done processing all images and annotations.\nNum of annotated cells: %d' % len(all_verts_list))
    print('Num of background patches: %d' % len(all_background_files))

    # Statistics about the annotated cells.
    dx_max, dy_max = 0, 0
    count, dx_mean, dy_mean = 0, 0, 0
    dx_list, dy_list = [], []
    for cell in all_verts_list:
        x_max = np.max(cell[:, 0])
        y_max = np.max(cell[:, 1])

        x_min = np.min(cell[:, 0])
        y_min = np.min(cell[:, 1])

        dx = x_max - x_min
        dy = y_max - y_min
        dx_list.append(dx)
        dy_list.append(dy)

        dx_max = max(dx_max, dx)
        dy_max = max(dy_max, dy)

        count += 1
        dx_mean += dx
        dy_mean += dy

    dx_mean = dx_mean / count
    dy_mean = dy_mean / count
    dx_mode = max(set(dx_list), key=dx_list.count)
    dy_mode = max(set(dy_list), key=dy_list.count)

    print('Max dx: %d, Max dy: %d' % (dx_max, dy_max))
    print('Mean dx: %d, Mean dy: %d' % (dx_mean, dy_mean))
    print('Mode dx: %d, Mode dy: %d' % (dx_mode, dy_mode))
    print('Standard deviation dx: %d, Standard deviation dy: %d' % (np.std(dx_list), np.std(dy_list)))

    ax = plt.subplot(1, 2, 1)
    ax.hist(dx_list, bins=100, label='dx')
    ax = plt.subplot(1, 2, 2)
    ax.hist(dy_list, bins=100, label='dy')
    save_path = '../data/MoNuSeg2018TrainData_patch_%dx%d/' % (patch_size, patch_size)
    plt.savefig(save_path + 'histogram.png')


    return

def process_MoNuSeg_Testdata(patch_size=32):
    folder = '../../external_data/MoNuSeg/MoNuSegTestData'

    annotation_files = sorted(glob(f'{folder}/*.xml'))
    image_files = sorted(glob(f'{folder}/*.tif'))
    #import pdb; pdb.set_trace()

    all_verts_list = []
    # e.g. TCGA-2Z-A9J9-01A-01-TS1.tif
    for i, annotation_file in enumerate(tqdm(annotation_files)):
        image_id = os.path.basename(annotation_file).split('.')[0]
        # debug
        # if image_id != 'TCGA-HE-7128-01Z-00-DX1':
        #     continue
        image_file = f'{folder}/{image_id}.tif'
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
        label, centroids_list = annotation_to_label(verts_list, image, image_id, region_id_list)

        # Divide the image and label into patches.
        patch_size = 32 # NOTE: Should be the same as the aug patch data.
        patches_folder = '../data/MoNuSeg2018TestData_patch_%dx%d/' % (patch_size, patch_size)
        patchify_and_save(image, image_id, label, centroids_list, patches_folder, patch_size)

    print('Done processing all images and annotations.\nAnnotated cells: %d' % len(all_verts_list))

    # Check the maximum size of the annotations.
    dx_max, dy_max = 0, 0
    count, dx_mean, dy_mean = 0, 0, 0
    dx_list, dy_list = [], []
    for cell in all_verts_list:
        x_max = np.max(cell[:, 0])
        y_max = np.max(cell[:, 1])

        x_min = np.min(cell[:, 0])
        y_min = np.min(cell[:, 1])

        dx = x_max - x_min
        dy = y_max - y_min
        dx_list.append(dx)
        dy_list.append(dy)

        dx_max = max(dx_max, dx)
        dy_max = max(dy_max, dy)

        count += 1
        dx_mean += dx
        dy_mean += dy

    dx_mean = dx_mean / count
    dy_mean = dy_mean / count
    dx_mode = max(set(dx_list), key=dx_list.count)
    dy_mode = max(set(dy_list), key=dy_list.count)

    print('Max dx: %d, Max dy: %d' % (dx_max, dy_max))
    print('Mean dx: %d, Mean dy: %d' % (dx_mean, dy_mean))
    print('Mode dx: %d, Mode dy: %d' % (dx_mode, dy_mode))
    print('Standard deviation dx: %d, Standard deviation dy: %d' % (np.std(dx_list), np.std(dy_list)))

    ax = plt.subplot(1, 2, 1)
    ax.hist(dx_list, bins=100, label='dx')
    ax = plt.subplot(1, 2, 2)
    ax.hist(dy_list, bins=100, label='dy')
    save_path = '../data/MoNuSeg2018TestData_patch_%dx%d/' % (patch_size, patch_size)
    plt.savefig(save_path + 'histogram.png')


def main(args):    
    # NOTE: This is very confusing that test data is patchified with a different patch size.
    # NOTE: but works for now.
    patch_size = args.patch_size
    aug_patch_size = args.aug_patch_size

    process_MoNuSeg_Traindata(patch_size, args.organ, args.background_ratio)
    #process_MoNuSeg_Testdata(aug_patch_size)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_size', type=int, default=96)
    parser.add_argument('--aug_patch_size', type=int, default=32)
    parser.add_argument('--background_ratio', type=float, default=0.3)
    parser.add_argument('--organ', type=str, default='Breast')
    args = parser.parse_args()

    main(args)
