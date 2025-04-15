import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from lacss.deploy import model_urls
from lacss.deploy.predict import Predictor
from csbdeep.utils import normalize


if __name__ == '__main__':

    # creates a pretrained model
    model = Predictor(model_urls['lacss3-base'])

    directory_list = sorted(glob('../results/MoNuSegByCancer/*/'))
    for directory in tqdm(directory_list):
        subset = directory.split('/')[-2]
        save_folder = os.path.join(directory, 'LACSS')
        os.makedirs(save_folder, exist_ok=True)

        image_list = glob(f'../../data/MoNuSeg/MoNuSegByCancer/{subset}/test/images/*.png')

        for image_path in image_list:
            image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            labels_pred = model.predict(normalize(image))['pred_label']

            mask_pred = np.uint8(labels_pred > 0) * 255
            mask_pred_path = os.path.join(save_folder, os.path.basename(image_path))
            cv2.imwrite(mask_pred_path, mask_pred)
