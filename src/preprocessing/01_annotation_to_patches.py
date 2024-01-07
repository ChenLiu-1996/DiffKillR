'''
Read annotations from json file, find the label maps, and patchify around them.
'''
import cv2
import os
import numpy as np
from typing import Tuple
from tqdm import tqdm
from skimage.draw import polygon2mask

class_value_map = {
    'EpithelialCell': 1,
    'EndothelialCell': 2,
    'Myocyte': 3,
    'Fibroblast': 4,
}


def load_vertices(json_path):
    '''
    Return a list of vertices for each polygon in the json file
    Each polygon is np.array of shape (n, 2), where n is the number of vertices
    '''
    import json
    with open(json_path) as f:
        data = json.load(f)

    class_verts_map = {}
    for cell_type in class_value_map.keys():

        # Find the vertex list for all annotations within the current cell type.
        feature_collection = data['features']
        cells = list(
            filter(
                lambda region: region["properties"]["classification"]["name"]
                == cell_type, feature_collection))

        verts_list = []
        for cell in cells:
            verts = cell["geometry"]["coordinates"]
            verts_list.append(np.array(verts).squeeze(0)[:, ::-1])

        class_verts_map[cell_type] = verts_list
    return class_verts_map


def annotation_to_label(class_verts_map: dict,
                        image: np.array) -> Tuple[np.array, dict]:
    label = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    class_centroid_map = {}
    for cell_type in class_value_map.keys():
        class_centroid_map[cell_type] = []

    for cell_type, cell_verts in class_verts_map.items():
        for cell in tqdm(cell_verts):
            cell_mask = polygon2mask(label.shape, cell).astype(
                np.uint8) * class_value_map[cell_type]
            label = np.maximum(label, cell_mask)

            centroid = np.argwhere(cell_mask > 0).sum(0) / (cell_mask
                                                            > 0).sum()
            class_centroid_map[cell_type].append(
                (int(centroid[0]), int(centroid[1])))

    return label, class_centroid_map


def patchify_and_save(image, label_map, class_centroid_map, patches_folder,
                      patch_size):
    for cell_type, centroid_list in class_centroid_map.items():
        for centroid in tqdm(centroid_list):
            patch_image_path = '%s/image/%s_H%d_W%d_patch_%dx%d.png' % (
                patches_folder, cell_type, centroid[0] - patch_size // 2,
                centroid[1] - patch_size // 2, patch_size, patch_size)
            patch_label_path = '%s/label/%s_H%d_W%d_patch_%dx%d.png' % (
                patches_folder, cell_type, centroid[0] - patch_size // 2,
                centroid[1] - patch_size // 2, patch_size, patch_size)
            patch_colored_label_path = '%s/colored_label/%s_H%d_W%d_patch_%dx%d.png' % (
                patches_folder, cell_type, centroid[0] - patch_size // 2,
                centroid[1] - patch_size // 2, patch_size, patch_size)
            os.makedirs(os.path.dirname(patch_image_path), exist_ok=True)
            os.makedirs(os.path.dirname(patch_label_path), exist_ok=True)
            os.makedirs(os.path.dirname(patch_colored_label_path),
                        exist_ok=True)

            h_begin = max(centroid[0] - patch_size // 2, 0)
            w_begin = max(centroid[1] - patch_size // 2, 0)
            h_end = min(h_begin + patch_size, image.shape[0])
            w_end = min(w_begin + patch_size, image.shape[1])

            # print('centroid', centroid)
            # print('h, w', h_begin, h_end, w_begin, w_end)

            patch_image = image[h_begin:h_end, w_begin:w_end, :]
            patch_label = label_map[h_begin:h_end, w_begin:w_end]

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
            patch_label_colored[patch_label == 2] = (0, 255, 0)
            patch_label_colored[patch_label == 3] = (255, 0, 0)
            patch_label_colored[patch_label == 4] = (255, 255, 255)
            cv2.imwrite(patch_colored_label_path, patch_label_colored)

    return


if __name__ == '__main__':
    patch_size = 96
    image_path = '../../raw_data/A28-87_CP_lvl1_HandE_1_Merged_RAW_ch00.tif'

    annotation_file = '../../raw_data/A28-87_CP_lvl1_HandE_1_Merged_RAW_ch00.tif_Annotations.json'
    patches_folder = '../../data/A28-87_CP_lvl1_HandE_1_Merged_RAW_ch00_patch_%dx%d/' % (
        patch_size, patch_size)

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    assert len(image.shape) == 3
    assert image.shape[-1] == 3

    # Read the annotation json.
    class_verts_map = load_vertices(annotation_file)
    for cell_type, cell_verts in class_verts_map.items():
        print('%d %s labeled.' % (len(cell_verts), cell_type))

    # Check the maximum size of the annotations.
    dx_max, dy_max = 0, 0
    count, dx_mean, dy_mean = 0, 0, 0
    for cell_type, cell_verts in class_verts_map.items():
        for cell in cell_verts:
            x_max = np.max(cell[:, 0])
            y_max = np.max(cell[:, 1])

            x_min = np.min(cell[:, 0])
            y_min = np.min(cell[:, 1])

            dx = x_max - x_min
            dy = y_max - y_min

            dx_max = max(dx_max, dx)
            dy_max = max(dy_max, dy)

            count += 1
            dx_mean += dx
            dy_mean += dy

    dx_mean = dx_mean / count
    dy_mean = dy_mean / count
    print('Max dx: %d, Max dy: %d' % (dx_max, dy_max))
    print('Mean dx: %d, Mean dy: %d' % (dx_mean, dy_mean))

    # Produce label from annotation.
    label_map, class_centroid_map = annotation_to_label(class_verts_map, image)

    # Divide the image and label into patches.
    patchify_and_save(image, label_map, class_centroid_map, patches_folder,
                      patch_size)
