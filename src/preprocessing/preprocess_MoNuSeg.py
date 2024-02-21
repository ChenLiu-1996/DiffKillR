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
            verts.append([y, x])
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
        #centroid = np.argwhere(cell_mask > 0).sum(0) / (cell_mask > 0).sum()
        centroids.append((int(centroid[0]), int(centroid[1])))

    return label, centroids


# def patchify_and_save(image, image_id, label, centroid_list,
#                       patches_folder, patch_size):
#     '''
#     Divide the image and label into patches and save them.

#     image: original image.
#     image_id: id of the image. This should be unique.
#     label: binary image mask.
#     centroid_list: list of centroids for each polygon/cell.

#     '''
#     for centroid in tqdm(centroid_list, desc='patchifying...'):
#         patch_image_path = '%s/image/%s_H%d_W%d_patch_%dx%d.png' % (
#             patches_folder, image_id, centroid[0] - patch_size // 2,
#             centroid[1] - patch_size // 2, patch_size, patch_size)
#         patch_label_path = '%s/label/%s_H%d_W%d_patch_%dx%d.png' % (
#             patches_folder, image_id, centroid[0] - patch_size // 2,
#             centroid[1] - patch_size // 2, patch_size, patch_size)
#         patch_colored_label_path = '%s/colored_label/%s_H%d_W%d_patch_%dx%d.png' % (
#                 patches_folder, image_id, centroid[0] - patch_size // 2,
#                 centroid[1] - patch_size // 2, patch_size, patch_size)
#         os.makedirs(os.path.dirname(patch_image_path), exist_ok=True)
#         os.makedirs(os.path.dirname(patch_label_path), exist_ok=True)
#         os.makedirs(os.path.dirname(patch_colored_label_path), exist_ok=True)


#         h_begin = max(centroid[0] - patch_size // 2, 0)
#         w_begin = max(centroid[1] - patch_size // 2, 0)
#         h_end = min(h_begin + patch_size, image.shape[0])
#         w_end = min(w_begin + patch_size, image.shape[1])

#         # print('centroid', centroid)
#         # print('h, w', h_begin, h_end, w_begin, w_end)

#         patch_image = image[h_begin:h_end, w_begin:w_end, :]
#         patch_label = label[h_begin:h_end, w_begin:w_end]

#         # Handle edge cases: literally on the edge.
#         if patch_image.shape != (patch_size, patch_size, 3):
#             h_diff = patch_size - patch_image.shape[0]
#             w_diff = patch_size - patch_image.shape[1]
#             patch_image = np.pad(patch_image,
#                                     pad_width=((0, h_diff), (0, w_diff), (0,
#                                                                         0)),
#                                     mode='constant')
#             patch_label = np.pad(patch_label,
#                                     pad_width=((0, h_diff), (0, w_diff)),
#                                     mode='constant')

#         patch_image = cv2.cvtColor(patch_image, cv2.COLOR_RGB2BGR)
#         cv2.imwrite(patch_image_path, patch_image)
#         cv2.imwrite(patch_label_path, patch_label)

#         # NOTE: Colored label only used for visual inspection!
#         patch_label_colored = np.zeros_like(patch_image)
#         patch_label_colored[patch_label == 1] = (0, 0, 255)
#         cv2.imwrite(patch_colored_label_path, patch_label_colored)

#     return

def process_MoNuSeg_data():
    '''
    images are in .tif format, RGB, 1000x1000.
    '''

    for subset in ['test', 'train']:

        if subset == 'train':
            image_folder = '../../external_data/MoNuSeg/MoNuSegTrainData/Tissue Images'
            annotation_folder = '../../external_data/MoNuSeg/MoNuSegTrainData/Annotations'

            out_image_folder = '../../external_data/MoNuSeg/MoNuSegTrainData/images/'
            out_mask_folder = '../../external_data/MoNuSeg/MoNuSegTrainData/masks/'

        else:
            image_folder = '../../external_data/MoNuSeg/MoNuSegTestData'
            annotation_folder = '../../external_data/MoNuSeg/MoNuSegTestData'

            out_image_folder = '../../external_data/MoNuSeg/MoNuSegTestData/images/'
            out_mask_folder = '../../external_data/MoNuSeg/MoNuSegTestData/masks/'

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
            mask, centroids_list = annotation_to_label(verts_list, image, image_id, region_id_list)

            os.makedirs(out_image_folder, exist_ok=True)
            os.makedirs(out_mask_folder, exist_ok=True)

            out_image_path = out_image_folder + '/' + image_id + '.png'
            out_mask_path = out_mask_folder + '/' + image_id + '.png'

            assert np.max(mask) in [0, 1]

            cv2.imwrite(out_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(out_mask_path, np.uint8(mask * 255))

    return

def subset_MoNuSeg_data_by_cancer():
    train_image_folder = '../../external_data/MoNuSeg/MoNuSegTrainData/images/'
    train_mask_folder = '../../external_data/MoNuSeg/MoNuSegTrainData/masks/'
    test_image_folder = '../../external_data/MoNuSeg/MoNuSegTestData/images/'
    test_mask_folder = '../../external_data/MoNuSeg/MoNuSegTestData/masks/'

    target_folder = '../../external_data/MoNuSeg/MoNuSegByCancer/'

    for cancer_type in ['breast', 'colon', 'prostate']:
        if cancer_type == 'breast':
            train_list = [
                'TCGA-A7-A13E-01Z-00-DX1',
                'TCGA-A7-A13F-01Z-00-DX1',
                'TCGA-AR-A1AK-01Z-00-DX1',
                'TCGA-AR-A1AS-01Z-00-DX1',
                'TCGA-E2-A1B5-01Z-00-DX1',
                'TCGA-E2-A14V-01Z-00-DX1',
            ]
            test_list = [
                'TCGA-AC-A2FO-01A-01-TS1',
                'TCGA-AO-A0J2-01A-01-BSA',
            ]
        if cancer_type == 'colon':
            train_list = [
                'TCGA-AY-A8YK-01A-01-TS1',
                'TCGA-NH-A8F7-01A-01-TS1',
            ]
            test_list = [
                'TCGA-A6-6782-01A-01-BS1'
            ]
        if cancer_type == 'prostate':
            train_list = [
                'TCGA-G9-6336-01Z-00-DX1',
                'TCGA-G9-6348-01Z-00-DX1',
                'TCGA-G9-6356-01Z-00-DX1',
                'TCGA-G9-6363-01Z-00-DX1',
                'TCGA-CH-5767-01Z-00-DX1',
                'TCGA-G9-6362-01Z-00-DX1'
            ]
            test_list = [
                'TCGA-EJ-A46H-01A-03-TSC',
                'TCGA-HC-7209-01A-01-TS1'
            ]

        for train_item in tqdm(train_list):
            image_path_from = train_image_folder + train_item + '.png'
            mask_path_from = train_mask_folder + train_item + '.png'
            image_path_to = target_folder + '/' + cancer_type + '/train/images/' + train_item + '.png'
            mask_path_to = target_folder + '/' + cancer_type + '/train/masks/' + train_item + '.png'

            os.makedirs(os.path.dirname(image_path_to), exist_ok=True)
            os.makedirs(os.path.dirname(mask_path_to), exist_ok=True)
            os.system('cp %s %s' % (image_path_from, image_path_to))
            os.system('cp %s %s' % (mask_path_from, mask_path_to))

        for test_item in tqdm(test_list):
            image_path_from = test_image_folder + test_item + '.png'
            mask_path_from = test_mask_folder + test_item + '.png'
            image_path_to = target_folder + '/' + cancer_type + '/test/images/' + test_item + '.png'
            mask_path_to = target_folder + '/' + cancer_type + '/test/masks/' + test_item + '.png'

            os.makedirs(os.path.dirname(image_path_to), exist_ok=True)
            os.makedirs(os.path.dirname(mask_path_to), exist_ok=True)
            os.system('cp %s %s' % (image_path_from, image_path_to))
            os.system('cp %s %s' % (mask_path_from, mask_path_to))

    return

def subset_patchify_MoNuSeg_data_by_cancer(imsize: int):
    train_image_folder = '../../external_data/MoNuSeg/MoNuSegTrainData/images/'
    train_mask_folder = '../../external_data/MoNuSeg/MoNuSegTrainData/masks/'
    test_image_folder = '../../external_data/MoNuSeg/MoNuSegTestData/images/'
    test_mask_folder = '../../external_data/MoNuSeg/MoNuSegTestData/masks/'

    target_folder = '../../external_data/MoNuSeg/MoNuSegByCancer_%sx%s/' % (imsize, imsize)

    for cancer_type in ['breast', 'colon', 'prostate']:
        if cancer_type == 'breast':
            train_list = [
                'TCGA-A7-A13E-01Z-00-DX1',
                'TCGA-A7-A13F-01Z-00-DX1',
                'TCGA-AR-A1AK-01Z-00-DX1',
                'TCGA-AR-A1AS-01Z-00-DX1',
                'TCGA-E2-A1B5-01Z-00-DX1',
                'TCGA-E2-A14V-01Z-00-DX1',
            ]
            test_list = [
                'TCGA-AC-A2FO-01A-01-TS1',
                'TCGA-AO-A0J2-01A-01-BSA',
            ]
        if cancer_type == 'colon':
            train_list = [
                'TCGA-AY-A8YK-01A-01-TS1',
                'TCGA-NH-A8F7-01A-01-TS1',
            ]
            test_list = [
                'TCGA-A6-6782-01A-01-BS1'
            ]
        if cancer_type == 'prostate':
            train_list = [
                'TCGA-G9-6336-01Z-00-DX1',
                'TCGA-G9-6348-01Z-00-DX1',
                'TCGA-G9-6356-01Z-00-DX1',
                'TCGA-G9-6363-01Z-00-DX1',
                'TCGA-CH-5767-01Z-00-DX1',
                'TCGA-G9-6362-01Z-00-DX1'
            ]
            test_list = [
                'TCGA-EJ-A46H-01A-03-TSC',
                'TCGA-HC-7209-01A-01-TS1'
            ]

        for train_item in tqdm(train_list):
            image_path_from = train_image_folder + train_item + '.png'
            mask_path_from = train_mask_folder + train_item + '.png'

            image = cv2.imread(image_path_from)
            mask = cv2.imread(mask_path_from)
            image_h, image_w = image.shape[:2]

            for h_chunk in range(image_h // imsize):
                for w_chunk in range(image_w // imsize):
                    h = h_chunk * imsize
                    w = w_chunk * imsize
                    image_path_to = target_folder + '/' + cancer_type + '/train/images/' + train_item + '_H%sW%s.png' % (h, w)
                    mask_path_to = target_folder + '/' + cancer_type + '/train/masks/' + train_item + '_H%sW%s.png' % (h, w)
                    os.makedirs(os.path.dirname(image_path_to), exist_ok=True)
                    os.makedirs(os.path.dirname(mask_path_to), exist_ok=True)

                    h_begin = max(h, 0)
                    w_begin = max(w, 0)
                    h_end = min(h + imsize, image_h)
                    w_end = min(w + imsize, image_w)

                    image_patch = image[h_begin:h_end, w_begin:w_end, :]
                    mask_patch = mask[h_begin:h_end, w_begin:w_end]

                    cv2.imwrite(image_path_to, image_patch)
                    cv2.imwrite(mask_path_to, mask_patch)

        for test_item in tqdm(test_list):
            image_path_from = test_image_folder + test_item + '.png'
            mask_path_from = test_mask_folder + test_item + '.png'

            image = cv2.imread(image_path_from)
            mask = cv2.imread(mask_path_from)
            image_h, image_w = image.shape[:2]

            for h_chunk in range(image_h // imsize):
                for w_chunk in range(image_w // imsize):
                    h = h_chunk * imsize
                    w = w_chunk * imsize

                    image_path_to = target_folder + '/' + cancer_type + '/test/images/' + test_item + '_H%sW%s.png' % (h, w)
                    mask_path_to = target_folder + '/' + cancer_type + '/test/masks/' + test_item + '_H%sW%s.png' % (h, w)
                    os.makedirs(os.path.dirname(image_path_to), exist_ok=True)
                    os.makedirs(os.path.dirname(mask_path_to), exist_ok=True)

                    h_begin = max(h, 0)
                    w_begin = max(w, 0)
                    h_end = min(h + imsize, image_h)
                    w_end = min(w + imsize, image_w)

                    image_patch = image[h_begin:h_end, w_begin:w_end, :]
                    mask_patch = mask[h_begin:h_end, w_begin:w_end]

                    cv2.imwrite(image_path_to, image_patch)
                    cv2.imwrite(mask_path_to, mask_patch)

    return

if __name__ == '__main__':
    process_MoNuSeg_data()
    subset_MoNuSeg_data_by_cancer()
    subset_patchify_MoNuSeg_data_by_cancer(imsize=200)
