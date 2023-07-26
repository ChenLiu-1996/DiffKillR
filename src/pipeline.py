from glob import glob
import os
import cv2
from tqdm import tqdm

from semantic_segmentation import apply_otsu_thresholding
from detection import detect_blob

if __name__ == '__main__':
    # Apply Otsu thresholding & LoG blob detection
    img_folder = '../data/'
    out_folder = '../blobs_data/'

    os.makedirs(out_folder, exist_ok=True)

    for img_path in tqdm(glob(img_folder + '*.jpg')):
        binarized_image, _ = apply_otsu_thresholding(img_path)
        blobs = detect_blob(binarized_image, min_sigma=3, max_sigma=50, num_sigma=30, threshold=0.12, overlap=0.5)

        # draw blobs
        og_image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        for blob in blobs:
            y, x, radius = blob
            #print(y,x,radius)
            cv2.circle(og_image, (int(x), int(y)), int(radius), (0, 255, 0), 2)
        
        # Save image
        blob_fname = os.path.basename(img_path).replace('.jpg', '_blobs.jpg')
        cv2.imwrite(out_folder + blob_fname, og_image)