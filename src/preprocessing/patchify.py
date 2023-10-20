from glob import glob
import os
from tqdm import tqdm
import numpy as np

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2  # import after setting OPENCV_IO_MAX_IMAGE_PIXELS

if __name__ == '__main__':
    image_path = sorted(glob('../../raw_data/*.tif'))[0]
    image_fname = os.path.basename(image_path).replace('.tif', '')
    out_shape = (64, 64, 3)
    out_folder = '../../data/%s_patch_H%sW%s/' % (image_fname, out_shape[0],
                                                  out_shape[1])
    os.makedirs(out_folder, exist_ok=True)

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    assert len(image.shape) == 3
    assert image.shape[-1] == 3

    nrows = int(np.ceil(image.shape[0] / out_shape[0]))
    ncols = int(np.ceil(image.shape[1] / out_shape[1]))

    for i in tqdm(range(nrows)):
        for j in range(ncols):
            h_pointer = i * out_shape[0]
            w_pointer = j * out_shape[1]

            h_end = h_pointer + out_shape[0]
            if h_end > image.shape[0]:
                h_end = None

            w_end = w_pointer + out_shape[1]
            if w_end > image.shape[1]:
                w_end = None

            patch = image[h_pointer:h_end, w_pointer:w_end]

            # Handle edge cases: literally on the edge.
            if patch.shape != out_shape:
                h_diff = out_shape[0] - patch.shape[0]
                w_diff = out_shape[1] - patch.shape[1]
                patch = np.pad(patch,
                               pad_width=((0, h_diff), (0, w_diff), (0, 0)),
                               mode='constant')

            # Do not save all-blank patches.
            if (patch == 0).all():
                continue

            # NOTE: This is a hot fix. Not sure what is a good heuristic.
            # Only save patches with significant BLUE channel.
            if patch[..., 2].max() < 180:
                continue
            patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
            output_path = out_folder + 'patch_H%sW%s.jpg' % (h_pointer,
                                                             w_pointer)

            cv2.imwrite(output_path, patch)
