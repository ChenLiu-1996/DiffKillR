'''
Read annotations,  Patchify 
images are in .tif format, RGB, 1000x1000.

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
import scipy.io as sio

def load_GLySAC_annotation(label_path: str):
    '''
    Load annotation from GLySAC dataset.
    Returns:
        binary_mask: np.array of shape (H, W)
        centroid_arr: np.array of shape (N, 2)
    '''

    mat = sio.loadmat(label_path)
    instance_mask = mat['inst_map']
    centroids = mat['inst_centroid'] # centroid[0] is width, centroid[1] is height
    # flip the centroid_arr to (height, width)
    centroid_arr = np.array([[c[1], c[0]] for c in centroids])
    print('centroid_arr shape: ', centroid_arr.shape)
    # float -> Int
    centroid_arr = np.int32(centroid_arr)
    binary_mask = np.uint8(instance_mask > 0)

    return binary_mask, centroid_arr


def compute_verts_from_centroids(binary_mask, centroid_arr):
    '''
    Compute the vertices of the bounding box of each cell.
    Returns:
        verts_list: list of np.array of shape (4, 2)
    '''
    verts_list = []
    for centroid in centroid_arr:
        x, y = centroid.astype(np.int32)
        label = binary_mask[x, y]
        if label == 0:
            print('Hmmm... centroid not in the cell.')
            print('checking mask[x,y]: ', binary_mask[x, y])
            print('checking mask[y,x]: ', binary_mask[y, x])
            continue

        # Find the top, bottom, right, left vertices.
        H, W = binary_mask.shape

        # topmost
        dx = 0
        while x + dx < H and binary_mask[x + dx, y] > 0:
            dx += 1
        top_vertex = (x + dx - 1, y)

        # bottommost
        dx = 0
        while x - dx >= 0 and binary_mask[x - dx, y] > 0:
            dx += 1
        bottom_vertex = (x - dx + 1, y)

        # rightmost
        dy = 0
        while y + dy < W and binary_mask[x, y + dy] > 0:
            dy += 1
        right_vertex = (x, y + dy - 1)

        # leftmost
        dy = 0
        while y - dy >= 0 and binary_mask[x, y - dy] > 0:
            dy += 1
        left_vertex = (x, y - dy + 1)
       
        verts = np.array([top_vertex, bottom_vertex, right_vertex, left_vertex]) # (4, 2)

        verts_list.append(verts)

    return verts_list


def patchify_and_save(image, image_id, label, centroid_list,
                      patches_folder, patch_size):
    '''
    Divide the image and label into patches and save them.

    image: original image.
    image_id: id of the image. This should be unique.
    label: binary image mask.
    centroid_list: centroids for each cell, shape (N, 2).
    patches_folder: folder to save the patches.
    patch_size: size of the patch.

    '''
    for centroid in tqdm(centroid_list, desc='patchifying...'):
        patch_image_path = '%s/image/%s_H%d_W%d_patch_%dx%d.png' % (
            patches_folder, image_id, centroid[0] - patch_size // 2,
            centroid[1] - patch_size // 2, patch_size, patch_size)
        patch_label_path = '%s/label/%s_H%d_W%d_patch_%dx%d.png' % (
            patches_folder, image_id, centroid[0] - patch_size // 2,
            centroid[1] - patch_size // 2, patch_size, patch_size)
        patch_colored_label_path = '%s/colored_label/%s_H%d_W%d_patch_%dx%d.png' % (
                patches_folder, image_id, centroid[0] - patch_size // 2,
                centroid[1] - patch_size // 2, patch_size, patch_size)
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

    return

def process_Traindata(patch_size=96):
    '''
        images are in .tif format, RGB, 1000x1000.
    '''
    image_folder = '../external_data/GLySAC/Train/Images'
    annotation_folder = '../external_data/GLySAC/Train/Labels'

    annotation_files = sorted(glob(f'{annotation_folder}/*.mat'))
    image_files = sorted(glob(f'{image_folder}/*.tif'))
    # import pdb; pdb.set_trace()

    all_verts_list = []

    for i, annotation_file in enumerate(tqdm(annotation_files)):
        image_id = os.path.basename(annotation_file).split('.')[0] # 'AGC1_tumor_2'
        image_file = f'{image_folder}/{image_id}.tif'
        if image_file not in image_files:
            print(f'Image file {image_file} not found.')
            continue

        image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        assert len(image.shape) == 3
        assert image.shape[-1] == 3

        # Read the annotation .mat file.
        binary_mask, centroids_arr = load_GLySAC_annotation(annotation_file)
        print('Done reading annotation for image %s' % image_id)
        print('Number of annotated cells: %d' % centroids_arr.shape[0])

        # topmost, bottommost rightmost, leftmost vertices of the centroids.
        verts_list = compute_verts_from_centroids(binary_mask, centroids_arr)

        all_verts_list.extend(verts_list)

        # Divide the image and label into patches.
        patches_folder = '../data/GLySACTrainData_patch_%dx%d/' % (patch_size, patch_size)
        # NOTE: make sure centroid_arr pixel of the mask is > 0;
        corrected_centroids_arr = []
        for c in centroids_arr:
            print('c: ', c)
            if binary_mask[c[0], c[1]] > 0:
                corrected_centroids_arr.append(c)
        
        print('Number of corrected centroids: ', len(corrected_centroids_arr))
        print('Number of incorrect centroids: ', len(centroids_arr) - len(corrected_centroids_arr))
        patchify_and_save(image, image_id, binary_mask, corrected_centroids_arr, patches_folder, patch_size)

    print('Done processing all images and annotations: annotated cells: %d' % len(all_verts_list))


    # Check the maximum size of the annotations.
    dx_max, dy_max = 0, 0
    count, dx_mean, dy_mean = 0, 0, 0
    dx_list, dy_list = [], []
    for cell in all_verts_list:
        assert cell.shape == (4, 2)

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

    import matplotlib.pyplot as plt
    ax = plt.subplot(1, 2, 1)
    ax.hist(dx_list, bins=100, label='dx')
    ax = plt.subplot(1, 2, 2)
    ax.hist(dy_list, bins=100, label='dy')
    save_path = '../data/GLySACTrainData_patch_%dx%d/' % (patch_size, patch_size)
    plt.savefig(save_path + 'histogram.png')


    return

def process_Testdata(patch_size=32):
    image_folder = '../external_data/GLySAC/Test/Images'
    annotation_folder = '../external_data/GLySAC/Test/Labels'

    annotation_files = sorted(glob(f'{annotation_folder}/*.mat'))
    image_files = sorted(glob(f'{image_folder}/*.tif'))
    #import pdb; pdb.set_trace()

    all_verts_list = []
    # e.g. TCGA-2Z-A9J9-01A-01-TS1.tif
    print('Number of annotation files: ', len(annotation_files))
    print('Number of image files: ', len(image_files))
    for i, annotation_file in enumerate(tqdm(annotation_files)):
        image_id = os.path.basename(annotation_file).split('.')[0]
        # debug
        # if image_id != 'TCGA-HE-7128-01Z-00-DX1':
        #     continue
        image_file = f'{image_folder}/{image_id}.tif'
        if image_file not in image_files:
            print(f'Image file {image_file} not found.')
            continue

        image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        assert len(image.shape) == 3
        assert image.shape[-1] == 3

        # Read the annotation .mat file.
        binary_mask, centroids_arr = load_GLySAC_annotation(annotation_file)
        print('Done reading annotation for image %s' % image_id)
        print('Number of annotated cells: %d' % centroids_arr.shape[0])

        # topmost, bottommost rightmost, leftmost vertices of the centroids.
        verts_list = compute_verts_from_centroids(binary_mask, centroids_arr)

        all_verts_list.extend(verts_list)

        # Divide the image and label into patches.
        patches_folder = '../data/GLySACTestData_patch_%dx%d/' % (patch_size, patch_size)
        # NOTE: make sure centroid_arr pixel of the mask is > 0;
        corrected_centroids_arr = []
        for c in centroids_arr:
            print('c: ', c)
            if binary_mask[c[0], c[1]] > 0:
                corrected_centroids_arr.append(c)
        
        print('Number of corrected centroids: ', len(corrected_centroids_arr))
        print('Number of incorrect centroids: ', len(centroids_arr) - len(corrected_centroids_arr))
        patchify_and_save(image, image_id, binary_mask, corrected_centroids_arr, patches_folder, patch_size)
    
    print('Done processing all images and annotations: annotated cells: %d' % len(all_verts_list))

    # Check the maximum size of the annotations.
    dx_max, dy_max = 0, 0
    count, dx_mean, dy_mean = 0, 0, 0
    dx_list, dy_list = [], []
    for cell in all_verts_list:
        assert cell.shape == (4, 2)

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

    import matplotlib.pyplot as plt
    ax = plt.subplot(1, 2, 1)
    ax.hist(dx_list, bins=100, label='dx')
    ax = plt.subplot(1, 2, 2)
    ax.hist(dy_list, bins=100, label='dy')
    save_path = '../data/GLySACTestData_patch_%dx%d/' % (patch_size, patch_size)
    plt.savefig(save_path + 'histogram.png')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_size', type=int, default=96)
    parser.add_argument('--aug_patch_size', type=int, default=32)
    args = parser.parse_args()

    # NOTE: This is very confusing that test data is patchified with a different patch size.
    # NOTE: but works for now.
    patch_size = args.patch_size
    aug_patch_size = args.aug_patch_size

    process_Traindata(patch_size)
    process_Testdata(aug_patch_size)
