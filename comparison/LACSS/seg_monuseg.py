import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from lacss.deploy import Predictor, model_urls
from csbdeep.utils import normalize


if __name__ == '__main__':

    # creates a pretrained model
    model = Predictor(model_urls['cnsp4-bf'])

    for folder in [
        'MoNuSegByCancer_200x200',
        'MoNuSegByCancer_intraimage5pct_200x200',
        'MoNuSegByCancer_intraimage20pct_200x200',
        'MoNuSegByCancer_intraimage50pct_200x200',
    ]:

        directory_list = sorted(glob('../results/%s/*/' % folder))
        for directory in tqdm(directory_list):
            subset = directory.split('/')[-2]
            save_folder = '%s/LACSS_stitched/' % directory
            os.makedirs(save_folder, exist_ok=True)

            if 'intraimage' in folder:
                cancer_type, img_id = subset.split('_')
                image_list = glob('../../external_data/MoNuSeg/%s/%s/%s_test/*_effective_image.png' % (folder, cancer_type, img_id))
            else:
                image_list = glob('../../external_data/MoNuSeg/MoNuSegByCancer/%s/test/images/*.png' % subset)

            for image_path in image_list:
                image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                labels_pred = model.predict(normalize(image))['pred_label']

                mask_pred = np.uint8(labels_pred > 0) * 255
                mask_pred_path = save_folder + os.path.basename(image_path).replace('_effective_image', '')
                cv2.imwrite(mask_pred_path, mask_pred)
