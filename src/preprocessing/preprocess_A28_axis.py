'''
Read annotations from json file, find the label maps, and patchify around them.
'''
import cv2
import os
import numpy as np
from typing import Tuple, List
from tqdm import tqdm
from skimage.draw import polygon2mask
from shapely.geometry import Point, Polygon


class_value_map = {
    'EpithelialCell': 1,
    'EndothelialCell': 2,
    'Myocyte': 3,
    'Fibroblast': 4,
}


def all_points_in_polygon(coords, polygon_coords) -> bool:
    polygon = Polygon(polygon_coords)
    return all(polygon.contains(Point(coord)) for coord in coords)


def load_cells_and_axis(json_path):
    '''
    Return a list of vertices for each polygon in the json file
    Each polygon is np.array of shape (n, 2), where n is the number of vertices
    '''
    import json
    with open(json_path) as f:
        data = json.load(f)

    data_type_list, data_cell_type_list, data_coord_list = [], [], []
    for item in data:
        data_type_list.append(item['geometry']['type'])
        try:
            data_cell_type_list.append(item['properties']['classification']['name'])
        except:
            data_cell_type_list.append(None)
        data_coord_list.append(item['geometry']['coordinates'])

    # Associate the axis annotations with the corresponding cells.
    cell_with_axis_coord_list = []

    # First collect the coordinates of all cells.
    cell_coords_list = []
    for i, data_type in enumerate(data_type_list):
        if data_type == 'Polygon':
            assert len(data_coord_list[i]) == 1
            cell_coords = data_coord_list[i][0]
            cell_coords_list.append((cell_coords, i))

    for i, data_type in enumerate(data_type_list):
        if data_type == 'LineString':
            # This is an axis annotation
            axis_coords = data_coord_list[i]

            # Find the cell containing the coordinates.
            cell_idx_list = []
            for cell_coords, cell_idx in cell_coords_list:
                if all_points_in_polygon(axis_coords, polygon_coords=cell_coords):
                    cell_idx_list.append(cell_idx)
            assert len(cell_idx_list) == 1
            cell_idx = cell_idx_list[0]

            assert data_cell_type_list[cell_idx] == 'EpithelialCell'

            cell_with_axis_coord_list.append(
                (np.array(data_coord_list[cell_idx]).squeeze(0)[:, ::-1],
                 np.array(axis_coords)[:, ::-1]))

    return cell_with_axis_coord_list


def apply_linear_gradient(label, direction):
    # Ensure the label is a binary mask (values are 0 or 1)
    label = np.clip(label, 0, 1)

    # Get the indices of the foreground elements
    foreground_indices = np.argwhere(label == 1)

    if foreground_indices.size == 0:
        return label.astype(np.uint8)

    # Calculate the direction vector
    direction = np.array(direction)
    norm_direction = direction / np.linalg.norm(direction)

    # Project the coordinates onto the direction vector
    projection = foreground_indices @ norm_direction

    # Normalize the projection to the range [0.1, 1]
    min_proj = np.min(projection)
    max_proj = np.max(projection)
    normalized_projection = 0.1 + 0.9 * (projection - min_proj) / (max_proj - min_proj) if max_proj != min_proj else np.zeros_like(projection)

    # Scale the normalized projection to the range [10, 255]
    gradient_values = (normalized_projection * 255).astype(np.uint8)

    # Create the gradient image
    gradient = np.zeros_like(label, dtype=np.uint8)
    gradient[foreground_indices[:, 0], foreground_indices[:, 1]] = gradient_values

    return gradient


def annotation_to_label(cell_with_axis_coord_list: List,
                        image: np.array) -> Tuple[np.array, dict]:
    binary_label = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    axis_label = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    centroid_list = []

    for cell_coords, axis_coords in tqdm(cell_with_axis_coord_list):
        cell_mask = polygon2mask(binary_label.shape, cell_coords).astype(
            np.uint8) * class_value_map['EpithelialCell']
        binary_label = np.maximum(binary_label, cell_mask)
        centroid = np.argwhere(cell_mask > 0).sum(0) / (cell_mask > 0).sum()
        centroid_list.append((int(centroid[0]), int(centroid[1])))

        direction = np.array(axis_coords)[0] - np.array(axis_coords)[1]

        cell_gradient = apply_linear_gradient(cell_mask, direction)
        axis_label = np.maximum(axis_label, cell_gradient)

    return binary_label, axis_label, centroid_list


def patchify_and_save(image,
                      binary_label,
                      axis_label,
                      centroid_list,
                      patches_folder,
                      patch_size):

    for centroid in tqdm(centroid_list):
        patch_image_path = '%s/image/%s_H%d_W%d_patch_%dx%d.png' % (
            patches_folder, 'EpithelialCell', centroid[0] - patch_size // 2,
            centroid[1] - patch_size // 2, patch_size, patch_size)
        patch_label_path = '%s/label/%s_H%d_W%d_patch_%dx%d.png' % (
            patches_folder, 'EpithelialCell', centroid[0] - patch_size // 2,
            centroid[1] - patch_size // 2, patch_size, patch_size)
        patch_axis_path = '%s/axis/%s_H%d_W%d_patch_%dx%d.png' % (
            patches_folder, 'EpithelialCell', centroid[0] - patch_size // 2,
            centroid[1] - patch_size // 2, patch_size, patch_size)
        patch_colored_label_path = '%s/colored_label/%s_H%d_W%d_patch_%dx%d.png' % (
            patches_folder, 'EpithelialCell', centroid[0] - patch_size // 2,
            centroid[1] - patch_size // 2, patch_size, patch_size)
        os.makedirs(os.path.dirname(patch_image_path), exist_ok=True)
        os.makedirs(os.path.dirname(patch_label_path), exist_ok=True)
        os.makedirs(os.path.dirname(patch_axis_path), exist_ok=True)
        os.makedirs(os.path.dirname(patch_colored_label_path), exist_ok=True)

        h_begin = max(0, int(np.round(centroid[0] - patch_size/2)))
        h_end = min(image.shape[0], int(np.round(centroid[0] + patch_size/2)))
        w_begin = max(0, int(np.round(centroid[1] - patch_size/2)))
        w_end = min(image.shape[1], int(np.round(centroid[1] + patch_size/2)))

        patch_image = image[h_begin:h_end, w_begin:w_end, :]
        patch_label = binary_label[h_begin:h_end, w_begin:w_end]
        patch_axis = axis_label[h_begin:h_end, w_begin:w_end]

        # Handle edge cases: literally on the edge.
        if patch_image.shape != (patch_size, patch_size, 3):
            h_diff = patch_size - patch_image.shape[0]
            w_diff = patch_size - patch_image.shape[1]
            patch_image = np.pad(patch_image,
                                 pad_width=((0, h_diff), (0, w_diff), (0, 0)),
                                 mode='constant')
            patch_label = np.pad(patch_label,
                                 pad_width=((0, h_diff), (0, w_diff)),
                                 mode='constant')
            patch_axis = np.pad(patch_axis,
                                pad_width=((0, h_diff), (0, w_diff)),
                                mode='constant')

        patch_image = cv2.cvtColor(patch_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(patch_image_path, patch_image)
        cv2.imwrite(patch_label_path, patch_label)
        cv2.imwrite(patch_axis_path, patch_axis)

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

    annotation_file = '../../raw_data/A28-87_CP_lvl1_HandE_1_Merged_RAW_ch00.tif_Annotations_axis.json'
    patches_folder = '../../data/A28-87_CP_lvl1_HandE_1_Merged_RAW_ch00_axis_patch_%dx%d/' % (
        patch_size, patch_size)

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    assert len(image.shape) == 3
    assert image.shape[-1] == 3

    # Read the annotation json.
    cell_with_axis_coord_list = load_cells_and_axis(annotation_file)

    # Produce label from annotation.
    binary_label, axis_label, centroid_list = annotation_to_label(cell_with_axis_coord_list, image)

    # Divide the image and label into patches.
    patchify_and_save(image, binary_label, axis_label, centroid_list, patches_folder, patch_size)
