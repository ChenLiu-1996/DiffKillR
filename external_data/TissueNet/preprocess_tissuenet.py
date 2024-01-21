import cv2
import os
import numpy as np
from tqdm import tqdm
import seaborn as sns
import warnings


def color_instances(label: np.array, palette: str = 'bright', n_colors: int = None) -> np.array:
    assert len(label.shape) == 2

    color_keys = np.unique(label)
    if color_keys[0] == 0:
        color_keys = color_keys[1:]
    else:
        warnings.warn('`color_instances`: color_keys[0] is not zero. It is %s instead.' % color_keys[0])

    if n_colors is None:
        n_colors = len(color_keys)
    color_values = sns.color_palette(palette, n_colors=n_colors)
    label_recolored = np.zeros((*label.shape, 3))

    for i, k in enumerate(color_keys):
        label_recolored[label==k, ...] = color_values[i]

    return label_recolored


def preprocess_tissuenet(npz_folder: str = './') -> None:
    npz_train = npz_folder + 'tissuenet_v1.1_train.npz'
    npz_val = npz_folder + 'tissuenet_v1.1_val.npz'
    npz_test = npz_folder + 'tissuenet_v1.1_test.npz'

    image_save_folder = npz_folder + 'images/'
    label_nuclei_save_folder = npz_folder + 'labels_nuclei/'
    label_cytoplasm_save_folder = npz_folder + 'labels_cytoplasm/'
    label_nuclei_colored_save_folder = npz_folder + 'labels_nuclei_colored/'
    label_cytoplasm_colored_save_folder = npz_folder + 'labels_cytoplasm_colored/'

    os.makedirs(image_save_folder, exist_ok=True)
    os.makedirs(label_nuclei_save_folder, exist_ok=True)
    os.makedirs(label_cytoplasm_save_folder, exist_ok=True)
    os.makedirs(label_nuclei_colored_save_folder, exist_ok=True)
    os.makedirs(label_cytoplasm_colored_save_folder, exist_ok=True)

    image_idx = 0
    for dataset in [npz_train, npz_val, npz_test]:
        with np.load(dataset, allow_pickle=True) as npdata:
            images = npdata['X']
            labels = npdata['y']
            metadata = npdata['meta']
        assert metadata[0][5] == 'specimen'

        for i in tqdm(range(len(images))):
            image_idx += 1
            image = images[i]
            label = labels[i]
            cell_type = metadata[i+1][5]

            image = cv2.resize(image, (256, 256))
            label = cv2.resize(label, (256, 256), interpolation=cv2.INTER_NEAREST)

            # Assign the image channels to "green" and "blue" colors.
            rgb_image = np.zeros((*image.shape[:2], 3))
            rgb_image[..., 1] = image[..., 0]
            rgb_image[..., 2] = image[..., 1]

            # Normalize to [0, 255] for saving
            rgb_image = np.float64(rgb_image)  # avoid overflow in the next step.
            rgb_image = 255 * rgb_image / np.percentile(rgb_image, 99)
            rgb_image[rgb_image > 255] = 255
            rgb_image = np.uint8(rgb_image)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

            nuclei = np.int16(label[..., 1])
            cytoplasm = np.int16(label[..., 0])
            # Use `n_colors` to maximize consistency of coloring for display purposes.
            # Though it barely helped...
            n_colors = len(np.unique(label))
            nuclei_colored = np.uint8(255 * color_instances(label=nuclei, n_colors=n_colors))
            cytoplasm_colored = np.uint8(255 * color_instances(label=cytoplasm, n_colors=n_colors))
            nuclei_colored = cv2.cvtColor(nuclei_colored, cv2.COLOR_RGB2BGR)
            cytoplasm_colored = cv2.cvtColor(cytoplasm_colored, cv2.COLOR_RGB2BGR)

            cv2.imwrite(image_save_folder + '%s_%s.png' % (str(image_idx).zfill(7), cell_type), rgb_image)
            cv2.imwrite(label_nuclei_save_folder + '%s_%s.png' % (str(image_idx).zfill(7), cell_type), nuclei)
            cv2.imwrite(label_cytoplasm_save_folder + '%s_%s.png' % (str(image_idx).zfill(7), cell_type), cytoplasm)
            cv2.imwrite(label_nuclei_colored_save_folder + '%s_%s.png' % (str(image_idx).zfill(7), cell_type), nuclei_colored)
            cv2.imwrite(label_cytoplasm_colored_save_folder + '%s_%s.png' % (str(image_idx).zfill(7), cell_type), cytoplasm_colored)


if __name__ == '__main__':
    preprocess_tissuenet()
