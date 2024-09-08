from segment_anything import SamPredictor, sam_model_registry

import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import sys
import numpy as np
from tqdm import tqdm
import skimage

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir + '/MedT/')

from utils import JointTransform2D, ImageToImage2D, Image2D
import cv2


parser = argparse.ArgumentParser(description='SAM')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--val_dataset', type=str)
parser.add_argument('--modelname', default='off', type=str,
                    help='name of the model to load')
parser.add_argument('--cuda', default="on", type=str,
                    help='switch on/off cuda option (default: off)')

parser.add_argument('--direc', default='./results', type=str,
                    help='directory to save')
parser.add_argument('--crop', type=int, default=None)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--loaddirec', default='load', type=str)
parser.add_argument('--imgsize', type=int, default=None)
parser.add_argument('--gray', default='no', type=str)
args = parser.parse_args()

direc = args.direc
gray_ = args.gray
modelname = args.modelname
imgsize = args.imgsize

if gray_ == "yes":
    from utils_gray import JointTransform2D, ImageToImage2D, Image2D
    imgchant = 1
else:
    from utils import JointTransform2D, ImageToImage2D, Image2D
    imgchant = 3

if args.crop is not None:
    crop = (args.crop, args.crop)
else:
    crop = None

tf_val = JointTransform2D(crop=crop, p_flip=0, color_jitter_params=None, long_mask=True)
val_dataset = ImageToImage2D(args.val_dataset, tf_val)
valloader = DataLoader(val_dataset, 1, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

sam = sam_model_registry["vit_b"](checkpoint=os.path.join('./checkpoints/', "medsam_vit_b.pth")).to(device)
model = SamPredictor(sam)

for batch_idx, (X_batch, y_batch, *rest) in enumerate(tqdm(valloader)):

    if isinstance(rest[0][0], str):
        image_filename = rest[0][0]
    else:
        image_filename = '%s.png' % str(batch_idx + 1).zfill(3)

    X_batch = X_batch.float().to(device)
    if X_batch.shape[1] == 1:
        X_batch = X_batch.repeat(1, 3, 1, 1)
    while len(y_batch.shape) < 4:
        y_batch = y_batch.unsqueeze(0)

    image = np.uint8(X_batch.permute(0, 2, 3, 1).squeeze(0).cpu().detach().numpy() * 255)
    label = y_batch.permute(0, 2, 3, 1).squeeze(0).squeeze(-1).cpu().detach().numpy()
    assert np.min(label) == 0

    # Resize to best size for MedSAM
    H, W = image.shape[:2]
    image = skimage.transform.resize(image, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.uint8)
    label = skimage.transform.resize(label, (1024, 1024), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)

    model.set_image(image)

    point_coords, point_label = [], []
    instance_label = skimage.measure.label(label)
    for instance_id in np.unique(instance_label)[1:]:
        instance_mask = (instance_label == instance_id).astype(int)
        # Following code block directly adapted from
        # https://github.com/mazurowski-lab/segment-anything-medical-evaluation/blob/main/prompt_gen_and_exec_v1.py
        padded_mask = np.uint8(np.pad(instance_mask, ((1, 1), (1, 1)), 'constant'))
        dist_img = cv2.distanceTransform(padded_mask,
                                         distanceType=cv2.DIST_L2,
                                         maskSize=5).astype(np.float32)[1:-1, 1:-1]
        # NOTE: numpy and opencv have inverse definition of row and column
        # NOTE: SAM and opencv have the same definition
        cY, cX = np.where(dist_img == dist_img.max())
        # NOTE: random seems to change DC by +/-1e-4
        # Random sample one point with largest distance
        random_idx = np.random.randint(0, len(cX))
        cX, cY = int(cX[random_idx]), int(cY[random_idx])

        # point: farthest from the object boundary
        point_coords.append((cX, cY))
        point_label.append(1)

    preds, _, _ = model.predict(point_coords=np.array(point_coords),
                                point_labels=np.array(point_label),
                                multimask_output=False)

    label_medsam = preds.squeeze(0)
    label_medsam = skimage.transform.resize(label_medsam, (H, W), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)

    assert len(label_medsam.shape) == 2
    label_medsam = label_medsam.astype(int)[None, None, ...]

    mask_true = y_batch.detach().cpu().numpy()
    mask_pred = label_medsam
    mask_pred[mask_pred>=0.5] = 1
    mask_pred[mask_pred<0.5] = 0
    mask_true[mask_true>0] = 1
    mask_true[mask_true<=0] = 0

    yval = (mask_true * 255).astype(int)
    yHaT = (mask_pred * 255).astype(int)

    del X_batch, y_batch, mask_pred, mask_true, label_medsam

    fulldir = direc+"/"

    if not os.path.isdir(fulldir):
        os.makedirs(fulldir)

    cv2.imwrite(fulldir+image_filename, yHaT[0,0,:,:])
