import torch
import argparse
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import numpy as np
import torch
from utils import JointTransform2D, ImageToImage2D, Image2D
import cv2
from tqdm import tqdm
import monai


parser = argparse.ArgumentParser(description='nnUNet')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run(default: 400)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=1, type=int,
                    metavar='N', help='batch size (default: 1)')
parser.add_argument('--learning_rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate (default: 0.001)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-5)')
parser.add_argument('--train_dataset', required=True, type=str)
parser.add_argument('--val_dataset', type=str)
parser.add_argument('--save_freq', type=int,default = 10)

parser.add_argument('--modelname', default='nnUNet', type=str,
                    help='type of model')
parser.add_argument('--cuda', default="on", type=str,
                    help='switch on/off cuda option (default: off)')
parser.add_argument('--aug', default='off', type=str,
                    help='turn on img augmentation (default: False)')
parser.add_argument('--load', default='default', type=str,
                    help='load a pretrained model')
parser.add_argument('--save', default='default', type=str,
                    help='save the model')
parser.add_argument('--direc', default='./nnUNet', type=str,
                    help='directory to save')
parser.add_argument('--crop', type=int, default=None)
parser.add_argument('--imgsize', type=int, default=None)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--gray', default='no', type=str)

args = parser.parse_args()
gray_ = args.gray
aug = args.aug
direc = args.direc
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

tf_train = JointTransform2D(crop=crop, p_flip=0.5, color_jitter_params=None, long_mask=True)
tf_val = JointTransform2D(crop=crop, p_flip=0, color_jitter_params=None, long_mask=True)
train_dataset = ImageToImage2D(args.train_dataset, tf_train)
val_dataset = ImageToImage2D(args.val_dataset, tf_val)
predict_dataset = Image2D(args.val_dataset)
dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valloader = DataLoader(val_dataset, 1, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

assert modelname == "nnUNet"

model = torch.nn.Sequential(
    monai.networks.nets.DynUNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=1,
        kernel_size=[5, 5, 5, 5],
        filters=[16, 32, 64, 128],
        strides=[1, 1, 1, 1],
        upsample_kernel_size=[1, 1, 1, 1]),
    torch.nn.Sigmoid()
)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model,device_ids=[0,1]).cuda()
model.to(device)

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(list(model.parameters()), lr=args.learning_rate,
                             weight_decay=1e-5)


pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total_params: {}".format(pytorch_total_params))

seed = 3000
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


for epoch in tqdm(range(args.epochs)):

    epoch_running_loss = 0

    for batch_idx, (X_batch, y_batch, *rest) in enumerate(dataloader):
        X_batch = Variable(X_batch.to(device))
        y_batch = Variable(y_batch.to(device))

        # ===================forward=====================
        output = model(X_batch).squeeze(1)
        loss = criterion(output, y_batch.float())

        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_running_loss += loss.item()

    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch, args.epochs, epoch_running_loss/(batch_idx+1)))

    if epoch == 10:
        for param in model.parameters():
            param.requires_grad = True

    if (epoch % args.save_freq) ==0:

        for batch_idx, (X_batch, y_batch, *rest) in enumerate(valloader):
            # print(batch_idx)
            if isinstance(rest[0][0], str):
                image_filename = rest[0][0]
            else:
                image_filename = '%s.png' % str(batch_idx + 1).zfill(3)

            X_batch = Variable(X_batch.to(device))
            y_batch = Variable(y_batch.to(device))

            y_out = model(X_batch)

            mask_true = y_batch.detach().cpu().numpy()
            mask_pred = y_out.detach().cpu().numpy()
            mask_pred[mask_pred>=0.5] = 1
            mask_pred[mask_pred<0.5] = 0
            mask_true[mask_true>0] = 1
            mask_true[mask_true<=0] = 0

            yval = (mask_true * 255).astype(int)
            yHaT = (mask_pred * 255).astype(int)

            del X_batch, y_batch, mask_pred, mask_true, y_out

            fulldir = direc+"/{}/".format(epoch)

            if not os.path.isdir(fulldir):
                os.makedirs(fulldir)

            cv2.imwrite(fulldir+image_filename, yHaT[0,0,:,:])

        fulldir = direc+"/{}/".format(epoch)
        torch.save(model.state_dict(), fulldir+args.modelname+".pth")
        torch.save(model.state_dict(), direc+"final_model.pth")



