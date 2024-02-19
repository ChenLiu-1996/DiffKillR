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
    for idx, cell in tqdm(enumerate(verts_list)):
        # cell is shape (n, 2)

        cell_mask = polygon2mask(label.shape, cell)
        label = np.maximum(label, cell_mask).astype(np.uint8)
        
        # if image_id == 'TCGA-HE-7128-01Z-00-DX1':
        #     print('cell', cell)
        #     print('idx: ', idx, 'region_id: ', region_id_list[idx])
        #     print('cell.shape', cell.shape)
            #print('cell_mask', cell_mask[:10, :10])

        centroid = skimage.measure.centroid(cell_mask)
        #centroid = np.argwhere(cell_mask > 0).sum(0) / (cell_mask > 0).sum()
        centroids.append((int(centroid[0]), int(centroid[1])))
        
    return label, centroids


def patchify_and_save(image, image_id, label, centroid_list, 
                      patches_folder, patch_size):
    '''
    Divide the image and label into patches and save them.
    
    image: original image.
    image_id: id of the image. This should be unique.
    label: binary image mask.
    centroid_list: list of centroids for each polygon/cell.

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

def process_MoNuSeg_data():
    '''
        images are in .tif format, RGB, 1000x1000.
    '''
    image_folder = '../../external_data/Chen_2024_MoNuSeg/MoNuSeg2018TrainData/Tissue Images'
    annotation_folder = '../../external_data/Chen_2024_MoNuSeg/MoNuSeg2018TrainData/Annotations'

    annotation_files = sorted(glob(f'{annotation_folder}/*.xml'))
    image_files = sorted(glob(f'{image_folder}/*.tif'))
    #import pdb; pdb.set_trace()

    all_verts_list = []
    
    for i, annotation_file in tqdm(enumerate(annotation_files)):
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

        # Read the annotation xml.
        verts_list, region_id_list = load_MoNuSeg_annotation(annotation_file)
        print('Done reading annotation for image %s' % image_id)
        print('Number of annotated cells: %d' % len(verts_list))
        all_verts_list.extend(verts_list)

        # Produce label from annotation for this image.
        label, centroids_list = annotation_to_label(verts_list, image, image_id, region_id_list)

        # Divide the image and label into patches.
        patch_size = 96 #TODO: dynamically set this.
        patches_folder = '../../data/MoNuSeg2018TrainData_patch_%dx%d/' % (patch_size, patch_size)
        patchify_and_save(image, image_id, label, centroids_list, patches_folder, patch_size)

    print('Done processing all images and annotations: annotated cells: %d' % len(all_verts_list))

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

    import matplotlib.pyplot as plt
    ax = plt.subplot(1, 2, 1)
    ax.hist(dx_list, bins=100, label='dx')
    ax = plt.subplot(1, 2, 2)
    ax.hist(dy_list, bins=100, label='dy')
    save_path = '../../data/MoNuSeg2018TrainData_patch_%dx%d/' % (patch_size, patch_size)
    plt.savefig(save_path + 'histogram.png')

    
    return

def process_test_MoNuSeg_data():
    folder = '../../external_data/Chen_2024_MoNuSeg/MoNuSegTestData'

    annotation_files = sorted(glob(f'{folder}/*.xml'))
    image_files = sorted(glob(f'{folder}/*.tif'))
    #import pdb; pdb.set_trace()

    all_verts_list = []
    # e.g. TCGA-2Z-A9J9-01A-01-TS1.tif
    for i, annotation_file in tqdm(enumerate(annotation_files)):
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
        patch_size = 32 # NOTE: Should be the same as the training patch data.
        patches_folder = '../../data/MoNuSeg2018TestData_patch_%dx%d/' % (patch_size, patch_size)
        patchify_and_save(image, image_id, label, centroids_list, patches_folder, patch_size)

    print('Done processing all images and annotations: annotated cells: %d' % len(all_verts_list))

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

    import matplotlib.pyplot as plt
    ax = plt.subplot(1, 2, 1)
    ax.hist(dx_list, bins=100, label='dx')
    ax = plt.subplot(1, 2, 2)
    ax.hist(dy_list, bins=100, label='dy')
    save_path = '../../data/MoNuSeg2018TestData_patch_%dx%d/' % (patch_size, patch_size)
    plt.savefig(save_path + 'histogram.png')

if __name__ == '__main__':
    #process_MoNuSeg_data()
    
    process_test_MoNuSeg_data()
    
