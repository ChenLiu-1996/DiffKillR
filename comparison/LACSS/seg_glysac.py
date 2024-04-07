import os
import cv2
import numpy as np
from glob import glob
from lacss.deploy import Predictor, model_urls
from csbdeep.utils import normalize



if __name__ == '__main__':

    # creates a pretrained model
    model = Predictor(model_urls['cnsp4-bf'])

    for dataset_name in ['tumor', 'normal']:
        save_folder = '../results/GLySACByTumor_200x200/%s/LACSS_stitched/' % dataset_name
        os.makedirs(save_folder, exist_ok=True)

        image_list = glob('../../external_data/GLySAC/GLySACByTumor/%s/test/images/*.png' % dataset_name)
        for image_path in image_list:
            image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            labels = model.predict(normalize(image))['pred_label']

            mask = np.uint8(labels > 0) * 255
            mask_path = save_folder + os.path.basename(image_path)
            cv2.imwrite(mask_path, mask)
