from typing import Dict, List, Tuple
import os
import sys
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import time


import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir + '/registration/')
from spatial_transformer import SpatialTransformer
from unet import UNet
from voxelmorph import VxmDense as VoxelMorph
from corrmlp import CorrMLP
from registration_utils import register_dipy, radially_color_mask_with_colormap, random_rectangle, random_triangle, random_star
from registration_loss import GradLoss as SmoothnessLoss

sys.path.insert(0, import_dir + '/utils/')
from metrics import dice_coeff, IoU, l1, ncc
from seed import seed_everything


def plot_predict_warp(fig, counter, moving_image, fixed_image, coeff_smoothness=0):
    model_list, time_list, image_l1_list, image_ncc_list, \
        mask_dice_list, mask_iou_list, diffeo_l1_list, diffeo_ncc_list = [], [], [], [], [], [], [], []

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    loss_fn_mse = torch.nn.MSELoss()
    loss_fn_smoothness = SmoothnessLoss('l2').loss

    _, diffeo_forward_dipy, _ = register_dipy(moving_image=moving_image, fixed_image=fixed_image)

    moving_image = radially_color_mask_with_colormap(moving_image)
    warper = SpatialTransformer(size=moving_image.shape[:2])
    warped_image_dipy = warper(torch.from_numpy(moving_image.transpose(2, 0, 1)[None, ...]).float(),
                          flow=torch.from_numpy(diffeo_forward_dipy.transpose(2, 0, 1)[None, ...]))
    warped_image_dipy = np.uint8(warped_image_dipy[0, ...]).transpose(1, 2, 0)

    # NOTE: For a fair comparison, starting from now, we will treat the
    # DiPy-warped image as the fixed image.
    fixed_image = warped_image_dipy

    ax = fig.add_subplot(6, 11, counter * 11 + 1)
    ax.imshow(moving_image, vmin=0, vmax=255)
    ax.set_axis_off()
    ax.set_title('Original Image', fontsize=24)

    ax = fig.add_subplot(6, 11, counter * 11 + 2)
    ax.imshow(warped_image_dipy, vmin=0, vmax=255)
    ax.set_axis_off()
    ax.set_title('Warped Image', fontsize=24)

    ax = fig.add_subplot(6, 11, counter * 11 + 3)
    vectors = [np.arange(0, s) for s in moving_image.shape[:2]]
    X, Y = np.meshgrid(vectors[0], vectors[1])
    warped_X = X + diffeo_forward_dipy[:, :, 1]
    warped_Y = Y + diffeo_forward_dipy[:, :, 0]
    for i in range(moving_image.shape[0]):
        ax.plot(warped_X[i, :], warped_Y[i, :], color='k')
    for j in range(moving_image.shape[1]):
        ax.plot(warped_X[:, j], warped_Y[:, j], color='k')
    ax.invert_yaxis()
    ax.set_axis_off()
    ax.set_title('Diffeomorphism (GT)', fontsize=24)


    # NOTE: DiffeoMappingNet with UNet.
    DiffeoMappingNet = UNet(
        num_filters=32,
        in_channels=6,
        out_channels=4)
    warped_image_unet, diffeo_forward_unet, time_unet = \
        train_eval_net(DiffeoMappingNet, device, moving_image, fixed_image, loss_fn_mse, loss_fn_smoothness, coeff_smoothness)

    model_list.append('UNet')
    time_list.append(time_unet)
    image_l1_list.append(l1(warped_image_unet, fixed_image))
    image_ncc_list.append(ncc(warped_image_unet, fixed_image))
    mask_dice_list.append(dice_coeff(warped_image_unet > 0, fixed_image > 0))
    mask_iou_list.append(IoU(warped_image_unet > 0, fixed_image > 0))
    diffeo_l1_list.append(l1(diffeo_forward_unet, diffeo_forward_dipy))
    diffeo_ncc_list.append(ncc(diffeo_forward_unet, diffeo_forward_dipy))

    ax = fig.add_subplot(6, 11, counter * 11 + 4)
    ax.imshow(warped_image_unet, vmin=0, vmax=255)
    ax.set_axis_off()
    ax.set_title(f'Predicted Warped Image\nDiffeoMappingNet (UNet)', fontsize=18)

    ax = fig.add_subplot(6, 11, counter * 11 + 5)
    vectors = [np.arange(0, s) for s in moving_image.shape[:2]]
    X, Y = np.meshgrid(vectors[0], vectors[1])
    warped_X = X + diffeo_forward_unet[:, :, 1]
    warped_Y = Y + diffeo_forward_unet[:, :, 0]
    for i in range(moving_image.shape[0]):
        ax.plot(warped_X[i, :], warped_Y[i, :], color='k')
    for j in range(moving_image.shape[1]):
        ax.plot(warped_X[:, j], warped_Y[:, j], color='k')
    ax.invert_yaxis()
    ax.set_axis_off()
    ax.set_title(f'Predicted Diffeomorphism\nDiffeoMappingNet (UNet)', fontsize=18)


    # NOTE: DiffeoMappingNet with VoxelMorph (non-diffeomorphic).
    DiffeoMappingNet = VoxelMorph(
        inshape=(64, 64),
        src_feats=3,
        trg_feats=3,
        int_steps=0,  # non-diffeomorphic
        bidir=True)

    warped_image_vm, diffeo_forward_vm, time_vm = \
        train_eval_net(DiffeoMappingNet, device, moving_image, fixed_image, loss_fn_mse, loss_fn_smoothness, coeff_smoothness)

    model_list.append('VM')
    time_list.append(time_vm)
    image_l1_list.append(l1(warped_image_vm, fixed_image))
    image_ncc_list.append(ncc(warped_image_vm, fixed_image))
    mask_dice_list.append(dice_coeff(warped_image_vm > 0, fixed_image > 0))
    mask_iou_list.append(IoU(warped_image_vm > 0, fixed_image > 0))
    diffeo_l1_list.append(l1(diffeo_forward_vm, diffeo_forward_dipy))
    diffeo_ncc_list.append(ncc(diffeo_forward_vm, diffeo_forward_dipy))

    ax = fig.add_subplot(6, 11, counter * 11 + 6)
    ax.imshow(warped_image_vm, vmin=0, vmax=255)
    ax.set_axis_off()
    ax.set_title(f'Predicted Warped Image\nDiffeoMappingNet (VM)', fontsize=18)

    ax = fig.add_subplot(6, 11, counter * 11 + 7)
    vectors = [np.arange(0, s) for s in moving_image.shape[:2]]
    X, Y = np.meshgrid(vectors[0], vectors[1])
    warped_X = X + diffeo_forward_vm[:, :, 1]
    warped_Y = Y + diffeo_forward_vm[:, :, 0]
    for i in range(moving_image.shape[0]):
        ax.plot(warped_X[i, :], warped_Y[i, :], color='k')
    for j in range(moving_image.shape[1]):
        ax.plot(warped_X[:, j], warped_Y[:, j], color='k')
    ax.invert_yaxis()
    ax.set_axis_off()
    ax.set_title(f'Predicted Diffeomorphism\nDiffeoMappingNet (VM)', fontsize=18)


    # NOTE: DiffeoMappingNet with VoxelMorph (diffeomorphic).
    DiffeoMappingNet = VoxelMorph(
        inshape=(64, 64),
        src_feats=3,
        trg_feats=3,
        bidir=True)
    warped_image_vmdiff, diffeo_forward_vmdiff, time_vmdiff = \
        train_eval_net(DiffeoMappingNet, device, moving_image, fixed_image, loss_fn_mse, loss_fn_smoothness, coeff_smoothness)

    model_list.append('VM-Diff')
    time_list.append(time_vmdiff)
    image_l1_list.append(l1(warped_image_vmdiff, fixed_image))
    image_ncc_list.append(ncc(warped_image_vmdiff, fixed_image))
    mask_dice_list.append(dice_coeff(warped_image_vmdiff > 0, fixed_image > 0))
    mask_iou_list.append(IoU(warped_image_vmdiff > 0, fixed_image > 0))
    diffeo_l1_list.append(l1(diffeo_forward_vmdiff, diffeo_forward_dipy))
    diffeo_ncc_list.append(ncc(diffeo_forward_vmdiff, diffeo_forward_dipy))

    ax = fig.add_subplot(6, 11, counter * 11 + 8)
    ax.imshow(warped_image_vmdiff, vmin=0, vmax=255)
    ax.set_axis_off()
    ax.set_title(f'Predicted Warped Image\nDiffeoMappingNet (VM-Diff)', fontsize=18)

    ax = fig.add_subplot(6, 11, counter * 11 + 9)
    vectors = [np.arange(0, s) for s in moving_image.shape[:2]]
    X, Y = np.meshgrid(vectors[0], vectors[1])
    warped_X = X + diffeo_forward_vmdiff[:, :, 1]
    warped_Y = Y + diffeo_forward_vmdiff[:, :, 0]
    for i in range(moving_image.shape[0]):
        ax.plot(warped_X[i, :], warped_Y[i, :], color='k')
    for j in range(moving_image.shape[1]):
        ax.plot(warped_X[:, j], warped_Y[:, j], color='k')
    ax.invert_yaxis()
    ax.set_axis_off()
    ax.set_title(f'Predicted Diffeomorphism\nDiffeoMappingNet (VM-Diff)', fontsize=18)

    # NOTE: DiffeoMappingNet with CorrMLP.
    DiffeoMappingNet = CorrMLP(
        in_channels=3,
    )
    warped_image_corrmlp, diffeo_forward_corrmlp, time_corrmlp = \
        train_eval_net(DiffeoMappingNet, device, moving_image, fixed_image, loss_fn_mse, loss_fn_smoothness, coeff_smoothness, learning_rate=1e-4)

    model_list.append('CorrMLP')
    time_list.append(time_corrmlp)
    image_l1_list.append(l1(warped_image_corrmlp, fixed_image))
    image_ncc_list.append(ncc(warped_image_corrmlp, fixed_image))
    mask_dice_list.append(dice_coeff(warped_image_corrmlp > 0, fixed_image > 0))
    mask_iou_list.append(IoU(warped_image_corrmlp > 0, fixed_image > 0))
    diffeo_l1_list.append(l1(diffeo_forward_corrmlp, diffeo_forward_dipy))
    diffeo_ncc_list.append(ncc(diffeo_forward_corrmlp, diffeo_forward_dipy))

    ax = fig.add_subplot(6, 11, counter * 11 + 10)
    ax.imshow(warped_image_corrmlp, vmin=0, vmax=255)
    ax.set_axis_off()
    ax.set_title(f'Predicted Warped Image\nDiffeoMappingNet (CorrMLP)', fontsize=18)

    ax = fig.add_subplot(6, 11, counter * 11 + 11)
    vectors = [np.arange(0, s) for s in moving_image.shape[:2]]
    X, Y = np.meshgrid(vectors[0], vectors[1])
    warped_X = X + diffeo_forward_corrmlp[:, :, 1]
    warped_Y = Y + diffeo_forward_corrmlp[:, :, 0]
    for i in range(moving_image.shape[0]):
        ax.plot(warped_X[i, :], warped_Y[i, :], color='k')
    for j in range(moving_image.shape[1]):
        ax.plot(warped_X[:, j], warped_Y[:, j], color='k')
    ax.invert_yaxis()
    ax.set_axis_off()
    ax.set_title(f'Predicted Diffeomorphism\nDiffeoMappingNet (CorrMLP)', fontsize=18)

    model_arr, time_arr, image_l1_arr, image_ncc_arr, mask_dice_arr, mask_iou_arr, diffeo_l1_arr, diffeo_ncc_arr = \
        list_to_np_arr(model_list, time_list, image_l1_list, image_ncc_list,
                       mask_dice_list, mask_iou_list, diffeo_l1_list, diffeo_ncc_list)
    result_dict = {
        'Architecture': model_arr,
        'Runtime': time_arr,
        'image L1': image_l1_arr,
        'image NCC': image_ncc_arr,
        'mask DSC': mask_dice_arr,
        'mask IoU': mask_iou_arr,
        'diffeo L1': diffeo_l1_arr,
        'diffeo NCC': diffeo_ncc_arr,
    }
    return result_dict

def list_to_np_arr(*lists: Tuple[List]) -> Tuple[np.array]:
    '''
    Some repetitive numpy casting of lists.
    '''
    return [np.array(_list) for _list in lists]

def merge_dict(*dicts: Tuple[Dict]) -> Dict:
    '''
    Merging dictionaries under my specific rule. Not generalizable.
    '''
    merged_dict = dicts[0]
    if len(dicts) > 1:
        for new_dict in dicts[1:]:
            for key in new_dict.keys():
                if key == 'Architecture':
                    pass
                else:
                    merged_dict[key] = np.vstack((merged_dict[key], new_dict[key]))
    return merged_dict

def dict_statistics(result_dict: Dict, digits: int = 3) -> None:
    '''
    Convert merged dict to dataframe and print statistics.
    '''
    df = pd.DataFrame(columns=result_dict.keys())

    for key in result_dict:
        if key == 'Architecture':
            df[key] = np.tile(result_dict[key], result_dict['Runtime'].shape[0])
        else:
            df[key] = result_dict[key].flatten('F')  # Flatten in Fortran-like (column-major) order

    df['Architecture'] = np.repeat(result_dict['Architecture'], result_dict['Runtime'].shape[0])
    df = df[['Architecture'] + [col for col in df.columns if col != 'Architecture']]
    df_mean = df.groupby('Architecture').mean().reset_index()
    df_std = df.groupby('Architecture').std().reset_index()

    df_latex = df_mean.copy()
    for metric in df.keys():
        if metric == 'Architecture':
            continue
        df_latex[metric] = df_mean[['Architecture', metric]].merge(
            df_std[['Architecture', metric]].rename(
                mapper={metric: metric + '_std'}, axis=1)).apply(
                lambda x: '$' + '{:.{}f}'.format(round(x[metric], digits), digits) + ' \pm ' + \
                    '{\color{gray} ' + '{:.{}f}'.format(round(x[metric + '_std'], digits), digits) + '}' + '$', axis = 1)

    df_latex.set_index('Architecture', inplace=True)
    df_latex = bold_best_per_row(df_latex.transpose())

    df_latex = df_latex[['UNet', 'VM', 'VM-Diff', 'CorrMLP']]
    print(df_latex.to_latex())
    return

def bold_best_per_row(df):

    column_name_list = df.columns.tolist()

    for row_idx, (key, row) in enumerate(df.iterrows()):
        values = [float(item.lstrip('$').split(' \\')[0]) for item in row.values]

        if key in ['Runtime', 'image L1', 'image L2', 'diffeo L1', 'diffeo L2']:
            best_val = np.min(values)
        else:
            best_val = np.max(values)
        best_loc_list = np.argwhere(values == best_val).flatten().tolist()

        for col_idx in best_loc_list:
            curr_str = df.iloc[row_idx, col_idx]
            new_str = curr_str
            new_str = new_str[0] + r'\textbf{' + new_str[1:]
            new_str_segments = new_str.split(' \\')
            assert len(new_str_segments) == 2
            new_str_segments[0] += '}'
            new_str = ' \\'.join(new_str_segments)

            df.at[key, column_name_list[col_idx]] = new_str

    return df

def train_eval_net(DiffeoMappingNet,
                   device,
                   moving_image,
                   fixed_image,
                   loss_fn_mse,
                   loss_fn_smoothness,
                   coeff_smoothness,
                   learning_rate: float = 1e-3):
    DiffeoMappingNet = DiffeoMappingNet.to(device)
    DiffeoMappingNet.train()
    warper = SpatialTransformer(size=moving_image.shape[:2])
    warper = warper.to(device)

    optimizer = torch.optim.AdamW(DiffeoMappingNet.parameters(), lr=learning_rate, weight_decay=1e-4)
    moving_image_torch = torch.from_numpy((moving_image).transpose(2, 0, 1)[None, ...]).float()
    fixed_image_torch = torch.from_numpy((fixed_image).transpose(2, 0, 1)[None, ...]).float()

    time_begin = time.time()
    for _ in tqdm(range(150)):
        __diffeo_forward, __diffeo_backward = DiffeoMappingNet(source=moving_image_torch, target=fixed_image_torch)
        __image_warped_forward = warper(moving_image_torch, flow=__diffeo_forward)
        __image_warped_backward = warper(fixed_image_torch, flow=__diffeo_backward)

        loss_forward = loss_fn_mse(fixed_image_torch, __image_warped_forward)
        loss_cyclic = loss_fn_mse(moving_image_torch, __image_warped_backward)
        loss = loss_forward + loss_cyclic
        if coeff_smoothness:
            loss_smoothness = loss_fn_smoothness(np.zeros((1)), __diffeo_forward) \
                            + loss_fn_smoothness(np.zeros((1)), __diffeo_backward)
            loss += coeff_smoothness * loss_smoothness

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    time_end = time.time()

    DiffeoMappingNet.eval()
    diffeo_forward_unet, _ = DiffeoMappingNet(source=moving_image_torch, target=fixed_image_torch)
    diffeo_forward_unet = diffeo_forward_unet.cpu().detach().numpy().squeeze(0).transpose(1, 2, 0)
    warped_image_unet = warper(torch.from_numpy(moving_image.transpose(2, 0, 1)[None, ...]).float(),
                               flow=torch.from_numpy(diffeo_forward_unet.transpose(2, 0, 1)[None, ...]))
    warped_image_unet = np.uint8(warped_image_unet[0, ...]).transpose(1, 2, 0)
    return warped_image_unet, diffeo_forward_unet, time_end - time_begin


if __name__ == '__main__':
    seed_everything(1)

    rectangle = random_rectangle(rectangle_size=(32, 32), center=(32, 32))
    star = random_star(center=(32, 32))
    triangle = random_triangle(center=(36, 32))

    # Plot figure.
    plt.rcParams["font.family"] = 'serif'
    n_cols, n_rows = 11, 6
    fig = plt.figure(figsize=(4*n_cols, 4*n_rows))

    result_dict = plot_predict_warp(fig, counter=0, moving_image=rectangle, fixed_image=star)
    result_dict = merge_dict(plot_predict_warp(fig, counter=1, moving_image=star, fixed_image=rectangle), result_dict)
    result_dict = merge_dict(plot_predict_warp(fig, counter=2, moving_image=star, fixed_image=triangle), result_dict)
    result_dict = merge_dict(plot_predict_warp(fig, counter=3, moving_image=triangle, fixed_image=star), result_dict)
    result_dict = merge_dict(plot_predict_warp(fig, counter=4, moving_image=triangle, fixed_image=rectangle), result_dict)
    result_dict = merge_dict(plot_predict_warp(fig, counter=5, moving_image=rectangle, fixed_image=triangle), result_dict)

    fig.tight_layout(pad=2)
    fig.savefig('predict_diffeomorphism.png', dpi=200)

    dict_statistics(result_dict, digits=2)
    dict_statistics(result_dict, digits=3)
