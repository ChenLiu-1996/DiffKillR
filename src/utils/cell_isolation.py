import cv2
import numpy as np
import torch
from simple_lama_inpainting import SimpleLama


__all__ = ['isolate_cell', 'nonzero_value_closest_to_center']


def isolate_cell(image: np.ndarray, label: np.ndarray, return_intermediates: bool = False, inpainting_model=None) -> np.ndarray:
    '''
    This function aims to clean the archetype cell image by removing any other cell in the region.

    Step 1. Find the location of archetype cell as well as the other cells.
            If only the archtype cell is present, do nothing and return image.
    Step 2. Mask out all cells.
    Step 3. Use image inpainting to fill the missing holes.
            Hopefully the resulting image will be all backgrounds that do not contain cells.
    Step 4. Add the archetype cell back.

    Assumptions:
    1. `image` contains at least 1 cell. The archetype cell will be at the center of the image.
    2. `label` gives non-zero values to cells. Different cells have different values.

    Documentation for SimpleLama(image, mask):
        image: 3 channel input image
        mask: 1 channel binary mask image where pixels with 255 will be inpainted.
    '''

    unique_indices = np.unique(label)
    num_cells = len(unique_indices) - 1 if 0 in unique_indices else len(unique_indices)
    if num_cells <= 1:
        if not return_intermediates:
            return image
        else:
            return None, None, None, image

    archetype_idx = nonzero_value_closest_to_center(label)
    all_cell_mask_255 = np.uint8(label > 0) * 255
    archetype_mask_255 = np.uint8(label == archetype_idx) * 255

    structure_element = np.ones((5, 5), np.uint8)
    all_cell_mask_255 = cv2.dilate(all_cell_mask_255, structure_element, iterations=4)
    archetype_mask_255 = cv2.dilate(archetype_mask_255, structure_element, iterations=1)

    if inpainting_model is None:
        inpainting_model = SimpleLama(device=torch.device('cpu'))

    with torch.no_grad():
        inpainted_image = np.array(inpainting_model(image, all_cell_mask_255))

    if return_intermediates:
        background_image = image.copy()
        background_image[all_cell_mask_255 > 0] = 0
        additional_output = [image, background_image, inpainted_image.copy()]

    cleaned_image = inpainted_image
    cleaned_image[archetype_mask_255 > 0] = image[archetype_mask_255 > 0]

    if not return_intermediates:
        return cleaned_image
    else:
        return *additional_output, cleaned_image


def nonzero_value_closest_to_center(label: np.ndarray) -> int:
    # Get the center of the image
    center = np.array(label.shape) / 2

    # Find indices of all non-zero elements
    non_zero_indices = np.argwhere(label != 0)

    # Compute the distances from the center
    distances = np.linalg.norm(non_zero_indices - center, axis=1)

    # Find the index of the closest non-zero element
    closest_index = np.argmin(distances)

    # Get the position and value of the closest non-zero element
    closest_position = non_zero_indices[closest_index]
    closest_value = label[tuple(closest_position)]
    return closest_value


if __name__ == '__main__':
    patch_id = 'TCGA-A7-A13E-01Z-00-DX1_H0W408_patch_96x96'

    image_path = f'../../data/MoNuSeg/MoNuSegByCancer_patch_96x96/Breast/train/images/{patch_id}.png'
    label_path = f'../../data/MoNuSeg/MoNuSegByCancer_patch_96x96/Breast/train/labels/{patch_id}.png'
    image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

    isolate_cell(image, label)
