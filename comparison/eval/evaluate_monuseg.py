import cv2
from glob import glob
import metrics

if __name__ == '__main__':

    for subset in ['breast', 'colon', 'prostate']:

        for model in ['PSM', 'MedT', 'UNet', 'nnUNet']:

            pred_folder = '../results/MoNuSegByCancer_200x200/%s/%s_stitched/' % (subset, model)
            true_folder = '../../external_data/MoNuSeg/MoNuSegByCancer/%s/test/masks/' % subset

            pred_list = sorted(glob(pred_folder + '*.png'))
            true_list = sorted(glob(true_folder + '*.png'))
            assert len(pred_list) == len(true_list)

            metric_list = []
            for pred_mask_path, true_mask_path in zip(pred_list, true_list):
                pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
                true_mask = cv2.imread(true_mask_path, cv2.IMREAD_GRAYSCALE)
                assert pred_mask.shape == true_mask.shape

                metric = metrics.compute_metrics(pred_mask, true_mask, ['p_F1', 'aji', 'iou'])
                metric_list.append(metric)

            print('MoNuSeg subset: %s Model: %s' % (subset, model))
            for key in metric_list[0].keys():
                num = sum([i[key] for i in metric_list]) / len(metric_list)
                print(F'{key}: {num}')
