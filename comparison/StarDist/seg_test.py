import os
import cv2
import numpy as np
from stardist.models import StarDist2D
from csbdeep.utils import normalize



if __name__ == '__main__':

    # creates a pretrained model
    # model = StarDist2D.from_pretrained('2D_versatile_he')
    # image_path = './EndothelialCell_H7247_W11558_patch_224x224.png'
    # image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    model = StarDist2D.from_pretrained('2D_paper_dsb2018')
    image_path = './0000001_Spleen.png'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    labels, _ = model.predict_instances(normalize(image))

    mask = np.uint8(labels > 0) * 255
    mask_path = os.path.basename(image_path).replace('.png', '_StarDist.png')
    cv2.imwrite(mask_path, mask)
