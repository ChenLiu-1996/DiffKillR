import cv2
from glob import glob
import metrics
import pandas as pd
import os


if __name__ == '__main__':

    for folder in [
        'MoNuSegByCancer',
        # 'MoNuSegByCancer_intraimage5pct_200x200',
        # 'MoNuSegByCancer_intraimage20pct_200x200',
        # 'MoNuSegByCancer_intraimage50pct_200x200',
    ]:

        directory_list = sorted(glob('../results/%s/*/' % folder))
        for directory in directory_list:
            subset = directory.split('/')[-2]

            # for model in ['UNet', 'nnUNet', 'MedT', 'LACSS', 'PSM', 'SAM', 'SAM2', 'SAM_Med2D', 'MedSAM', 'Ours_gt_loc', 'Ours']:
            for model in ['DiffKillR_cellIsolation-True', 'DiffKillR_cellIsolation-False', 'DiffKillR_gt_loc_cellIsolation-True', 'DiffKillR_gt_loc_cellIsolation-False']:
                for seed in range(1, 4):
                    pred_folder = '%s/%s_seed%d/' % (directory, model, seed)

                    if model in ['LACSS', 'SAM', 'SAM2', 'SAM_Med2D', 'MedSAM']:
                        if seed > 1:
                            continue
                        pred_folder = '%s/%s/' % (directory, model)
                    pred_list = sorted(glob(pred_folder + '*.png'))

                    print('>>> Working on: MoNuSeg [%s] Model [%s] seed %d' % (directory, model, seed))

                    true_folder = '../../data/MoNuSeg/MoNuSegByCancer/%s/test/masks/' % subset
                    true_list = sorted(glob(true_folder + '*.png'))

                    if len(pred_list) != len(true_list):
                        continue

                    print('> Found: MoNuSeg [%s] Model [%s] seed %d' % (directory, model, seed))

                    metric_list = []
                    for pred_mask_path, true_mask_path in zip(pred_list, true_list):
                        pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
                        true_mask = cv2.imread(true_mask_path, cv2.IMREAD_GRAYSCALE)
                        assert pred_mask.shape == true_mask.shape

                        metric = metrics.compute_metrics(pred_mask, true_mask, ['p_F1', 'aji', 'iou'])
                        metric_list.append(metric)

                    Dice = sum([i['dice'] for i in metric_list]) / len(metric_list)
                    IoU = sum([i['iou'] for i in metric_list]) / len(metric_list)
                    F1 = sum([i['p_F1'] for i in metric_list]) / len(metric_list)
                    AJI = sum([i['aji'] for i in metric_list]) / len(metric_list)

                    print('Dice: %.2f, IoU: %.2f, F1: %.2f, AJI: %.2f'
                        % (Dice, IoU, F1, AJI))

                    results_list = [[folder, subset, model, seed, true_folder, pred_folder, Dice, IoU, F1, AJI]]
                    results_df = pd.DataFrame(results_list, columns=['folder', 'subset', 'model', 'seed', 'GT_folder', 'pred_folder', 'Dice', 'IoU', 'F1', 'AJI'])

                    if os.path.isfile('./results_monuseg.csv'):
                        results_df.to_csv('./results_monuseg.csv', mode='a', header=False)
                    else:
                        results_df.to_csv('./results_monuseg.csv')
