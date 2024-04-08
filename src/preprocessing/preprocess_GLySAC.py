'''
Read annotations from xml file, find the label maps, and patchify around them.
images are in .tif format, RGB, 1000x1000.

'''
import cv2
import os
import numpy as np
from tqdm import tqdm
from glob import glob
import scipy.io


def load_GLySAC_annotation(mat_path: str) -> list[np.ndarray]:
    '''
    Return the instance segmentation mask and centroid array.
    '''
    mat = scipy.io.loadmat(mat_path)

    instance_mask = mat['inst_map']
    centroid_arr = mat['inst_centroid']
    binary_mask = np.uint8(instance_mask > 0) * 255

    return binary_mask, centroid_arr


def process_GLySAC_data():
    '''
    images are in .tif format, RGB, 1000x1000.
    '''

    for subset in ['test', 'train']:

        if subset == 'train':
            image_folder = '../../external_data/GLySAC/Train/Images'
            annotation_folder = '../../external_data/GLySAC/Train/Labels'

            out_image_folder = '../../external_data/GLySAC/Train/images'
            out_mask_folder = '../../external_data/GLySAC/Train/masks'

        else:
            image_folder = '../../external_data/GLySAC/Test/Images'
            annotation_folder = '../../external_data/GLySAC/Test/Labels'

            out_image_folder = '../../external_data/GLySAC/Test/images'
            out_mask_folder = '../../external_data/GLySAC/Test/masks'

        annotation_files = sorted(glob(f'{annotation_folder}/*.mat'))
        image_files = sorted(glob(f'{image_folder}/*.tif'))

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

            # Read the annotation mat.
            mask, centroid_arr = load_GLySAC_annotation(annotation_file)
            print('Done reading annotation for image %s' % image_id)
            print('Number of annotated cells: %d' % len(centroid_arr))

            os.makedirs(out_image_folder, exist_ok=True)
            os.makedirs(out_mask_folder, exist_ok=True)

            out_image_path = out_image_folder + '/' + image_id + '.png'
            out_mask_path = out_mask_folder + '/' + image_id + '.png'

            cv2.imwrite(out_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(out_mask_path, mask)

    return

def subset_GLySAC_data_by_tumor():
    train_image_folder = '../../external_data/GLySAC/Train/images/'
    train_mask_folder = '../../external_data/GLySAC/Train/masks/'
    test_image_folder = '../../external_data/GLySAC/Test/images/'
    test_mask_folder = '../../external_data/GLySAC/Test/masks/'

    target_folder = '../../external_data/GLySAC/GLySACByTumor/'

    for tumor_type in ['tumor', 'normal']:
        if tumor_type == 'tumor':
            train_list = [
                'AGC1_tumor_1',
                'AGC1_tumor_3',
                'AGC1_tumor_5',
                'AGC1_tumor_7',
                'AGC1_tumor_8',
                'AGC1_tumor_9',
                'AGC1_tumor_10',
                'DB-0001_tumor_1',
                'DB-0001_tumor_3',
                'DB-0003_tumor_1',
                'DB-0003_tumor_2',
                'DB-0466_tumor_2',
                'DB-0466_tumor_3',
            ]
            test_list = [
                'AGC1_tumor_2',
                'AGC1_tumor_4',
                'AGC1_tumor_11',
                'DB-0001_tumor_2',
                'DB-0466_tumor_1',
                'EGC1_new_tumor_1',
                'EGC1_new_tumor_2',
                'EGC1_new_tumor_3',
                'EGC1_new_tumor_4',
                'EGC1_new_tumor_5',
                'EGC1_new_tumor_6',
                'EGC1_new_tumor_7',
                'EGC1_new_tumor_10',
                'EGC1_new_tumor_11',
            ]
        if tumor_type == 'normal':
            train_list = [
                'DB-0001_normal_2',
                'DB-0003_normal_1',
                'DB-0466_normal_1',
                'DB-0466_normal_2',
                'DB-0466_normal_3',
                'EGC1_new_normal_1',
                'EGC1_new_normal_3',
            ]
            test_list = [
                'DB-0001_normal_1',
                'DB-0037_normal_1',
                'EGC1_new_normal_2',
                'EGC1_new_normal_5',
            ]

        for train_item in tqdm(train_list):
            image_path_from = train_image_folder + train_item + '.png'
            mask_path_from = train_mask_folder + train_item + '.png'
            image_path_to = target_folder + '/' + tumor_type + '/train/images/' + train_item + '.png'
            mask_path_to = target_folder + '/' + tumor_type + '/train/masks/' + train_item + '.png'

            os.makedirs(os.path.dirname(image_path_to), exist_ok=True)
            os.makedirs(os.path.dirname(mask_path_to), exist_ok=True)
            os.system('cp %s %s' % (image_path_from, image_path_to))
            os.system('cp %s %s' % (mask_path_from, mask_path_to))

        for test_item in tqdm(test_list):
            image_path_from = test_image_folder + test_item + '.png'
            mask_path_from = test_mask_folder + test_item + '.png'
            image_path_to = target_folder + '/' + tumor_type + '/test/images/' + test_item + '.png'
            mask_path_to = target_folder + '/' + tumor_type + '/test/masks/' + test_item + '.png'

            os.makedirs(os.path.dirname(image_path_to), exist_ok=True)
            os.makedirs(os.path.dirname(mask_path_to), exist_ok=True)
            os.system('cp %s %s' % (image_path_from, image_path_to))
            os.system('cp %s %s' % (mask_path_from, mask_path_to))

    return

def subset_patchify_GLySAC_data_by_tumor(imsize: int):
    train_image_folder = '../../external_data/GLySAC/Train/images/'
    train_mask_folder = '../../external_data/GLySAC/Train/masks/'
    test_image_folder = '../../external_data/GLySAC/Test/images/'
    test_mask_folder = '../../external_data/GLySAC/Test/masks/'

    target_folder = '../../external_data/GLySAC/GLySACByTumor_%sx%s/' % (imsize, imsize)

    for tumor_type in ['tumor', 'normal']:
        if tumor_type == 'tumor':
            train_list = [
                'AGC1_tumor_1',
                'AGC1_tumor_3',
                'AGC1_tumor_5',
                'AGC1_tumor_7',
                'AGC1_tumor_8',
                'AGC1_tumor_9',
                'AGC1_tumor_10',
                'DB-0001_tumor_1',
                'DB-0001_tumor_3',
                'DB-0003_tumor_1',
                'DB-0003_tumor_2',
                'DB-0466_tumor_2',
                'DB-0466_tumor_3',
            ]
            test_list = [
                'AGC1_tumor_2',
                'AGC1_tumor_4',
                'AGC1_tumor_11',
                'DB-0001_tumor_2',
                'DB-0466_tumor_1',
                'EGC1_new_tumor_1',
                'EGC1_new_tumor_2',
                'EGC1_new_tumor_3',
                'EGC1_new_tumor_4',
                'EGC1_new_tumor_5',
                'EGC1_new_tumor_6',
                'EGC1_new_tumor_7',
                'EGC1_new_tumor_10',
                'EGC1_new_tumor_11',
            ]
        if tumor_type == 'normal':
            train_list = [
                'DB-0001_normal_2',
                'DB-0003_normal_1',
                'DB-0466_normal_1',
                'DB-0466_normal_2',
                'DB-0466_normal_3',
                'EGC1_new_normal_1',
                'EGC1_new_normal_3',
            ]
            test_list = [
                'DB-0001_normal_1',
                'DB-0037_normal_1',
                'EGC1_new_normal_2',
                'EGC1_new_normal_5',
            ]

        for train_item in tqdm(train_list):
            image_path_from = train_image_folder + train_item + '.png'
            mask_path_from = train_mask_folder + train_item + '.png'

            image = cv2.imread(image_path_from)
            mask = cv2.imread(mask_path_from, cv2.IMREAD_UNCHANGED)
            image_h, image_w = image.shape[:2]

            for h_chunk in range(image_h // imsize):
                for w_chunk in range(image_w // imsize):
                    h = h_chunk * imsize
                    w = w_chunk * imsize
                    image_path_to = target_folder + '/' + tumor_type + '/train/images/' + train_item + '_H%sW%s.png' % (h, w)
                    mask_path_to = target_folder + '/' + tumor_type + '/train/masks/' + train_item + '_H%sW%s.png' % (h, w)
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
            mask = cv2.imread(mask_path_from, cv2.IMREAD_UNCHANGED)
            image_h, image_w = image.shape[:2]

            for h_chunk in range(image_h // imsize):
                for w_chunk in range(image_w // imsize):
                    h = h_chunk * imsize
                    w = w_chunk * imsize

                    image_path_to = target_folder + '/' + tumor_type + '/test/images/' + test_item + '_H%sW%s.png' % (h, w)
                    mask_path_to = target_folder + '/' + tumor_type + '/test/masks/' + test_item + '_H%sW%s.png' % (h, w)
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

def subset_patchify_GLySAC_data_by_tumor_intraimage(imsize: int):
    test_image_folder = '../../external_data/GLySAC/Test/images/'
    test_mask_folder = '../../external_data/GLySAC/Test/masks/'

    for tumor_type in ['tumor', 'normal']:
        if tumor_type == 'tumor':
            test_list = [
                'AGC1_tumor_2',
                'AGC1_tumor_4',
                'AGC1_tumor_11',
                'DB-0001_tumor_2',
                'DB-0466_tumor_1',
                'EGC1_new_tumor_1',
                'EGC1_new_tumor_2',
                'EGC1_new_tumor_3',
                'EGC1_new_tumor_4',
                'EGC1_new_tumor_5',
                'EGC1_new_tumor_6',
                'EGC1_new_tumor_7',
                'EGC1_new_tumor_10',
                'EGC1_new_tumor_11',
            ]
        if tumor_type == 'normal':
            test_list = [
                'DB-0001_normal_1',
                'DB-0037_normal_1',
                'EGC1_new_normal_2',
                'EGC1_new_normal_5',
            ]

        for percentage in [5, 20, 50]:
            target_folder = '../../external_data/GLySAC/GLySACByTumor_intraimage%dpct_%sx%s/' % (percentage, imsize, imsize)

            for test_item_count, test_item in enumerate(tqdm(test_list)):
                image_path_from = test_image_folder + test_item + '.png'
                mask_path_from = test_mask_folder + test_item + '.png'

                image = cv2.imread(image_path_from)
                mask = cv2.imread(mask_path_from)
                image_h, image_w = image.shape[:2]

                total_count = (image_h // imsize) * (image_w // imsize)
                target_count = int(np.ceil(percentage * total_count / 100))
                curr_count = 0

                for h_chunk in range(image_h // imsize):
                    for w_chunk in range(image_w // imsize):
                        h = h_chunk * imsize
                        w = w_chunk * imsize

                        h_begin = max(h, 0)
                        w_begin = max(w, 0)
                        h_end = min(h + imsize, image_h)
                        w_end = min(w + imsize, image_w)

                        image_patch = image[h_begin:h_end, w_begin:w_end, :]
                        mask_patch = mask[h_begin:h_end, w_begin:w_end]

                        if curr_count < target_count:
                            # 1. Save the image/mask pair to the train folder.
                            image_path_to = target_folder + '/' + tumor_type + \
                                '/img%d_train/images/' % test_item_count + test_item + '_H%sW%s.png' % (h, w)
                            mask_path_to = target_folder + '/' + tumor_type + \
                                '/img%d_train/masks/' % test_item_count + test_item + '_H%sW%s.png' % (h, w)
                            os.makedirs(os.path.dirname(image_path_to), exist_ok=True)
                            os.makedirs(os.path.dirname(mask_path_to), exist_ok=True)

                            cv2.imwrite(image_path_to, image_patch)
                            cv2.imwrite(mask_path_to, mask_patch)

                            # 2. Save an empty image/mask pair to the test folder.
                            empty_image_patch = image_patch * 0
                            empty_mask_patch = mask_patch * 0
                            empty_image_path_to = target_folder + '/' + tumor_type + \
                                '/img%d_test/images/' % test_item_count + test_item + '_H%sW%s.png' % (h, w)
                            empty_mask_path_to = target_folder + '/' + tumor_type + \
                                '/img%d_test/masks/' % test_item_count + test_item + '_H%sW%s.png' % (h, w)
                            os.makedirs(os.path.dirname(empty_image_path_to), exist_ok=True)
                            os.makedirs(os.path.dirname(empty_mask_path_to), exist_ok=True)

                            cv2.imwrite(empty_image_path_to, empty_image_patch)
                            cv2.imwrite(empty_mask_path_to, empty_mask_patch)

                        else:
                            # Save the image/mask pair to the test folder.
                            image_path_to = target_folder + '/' + tumor_type + \
                                '/img%d_test/images/' % test_item_count + test_item + '_H%sW%s.png' % (h, w)
                            mask_path_to = target_folder + '/' + tumor_type + \
                                '/img%d_test/masks/' % test_item_count + test_item + '_H%sW%s.png' % (h, w)
                            os.makedirs(os.path.dirname(image_path_to), exist_ok=True)
                            os.makedirs(os.path.dirname(mask_path_to), exist_ok=True)

                            cv2.imwrite(image_path_to, image_patch)
                            cv2.imwrite(mask_path_to, mask_patch)

                        curr_count += 1
    return

if __name__ == '__main__':
    process_GLySAC_data()
    subset_GLySAC_data_by_tumor()
    subset_patchify_GLySAC_data_by_tumor(imsize=200)
    subset_patchify_GLySAC_data_by_tumor_intraimage(imsize=200)
