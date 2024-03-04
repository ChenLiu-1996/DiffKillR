import os
import cv2
import numpy as np
from glob import glob
from stardist.models import StarDist2D
from csbdeep.utils import normalize



if __name__ == '__main__':

    # prints a list of available models
    # StarDist2D.from_pretrained()

    # creates a pretrained model
    model = StarDist2D.from_pretrained('2D_versatile_he')

    for dataset_name in ['breast', 'colon', 'prostate']:
        save_folder = '../results/MoNuSegByCancer_200x200/%s/StarDist_stitched/' % dataset_name
        os.makedirs(save_folder, exist_ok=True)

        image_list = glob('../../external_data/MoNuSeg/MoNuSegByCancer/%s/test/images/*.png' % dataset_name)
        for image_path in image_list:
            image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            labels, _ = model.predict_instances(normalize(image))

            mask = np.uint8(labels > 0) * 255
            mask_path = save_folder + os.path.basename(image_path)
            cv2.imwrite(mask_path, mask)
