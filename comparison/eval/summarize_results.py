import pandas as pd
import numpy as np


def get_folder_list(dataset: str):
    if dataset == 'glysac':
        folder_list = [
            'GLySACByTumor_200x200',
            'GLySACByTumor_intraimage5pct_200x200',
            'GLySACByTumor_intraimage20pct_200x200',
            'GLySACByTumor_intraimage50pct_200x200',
        ]
    elif dataset == 'monuseg':
        folder_list = [
            'MoNuSegByCancer_200x200',
            'MoNuSegByCancer_intraimage5pct_200x200',
            'MoNuSegByCancer_intraimage20pct_200x200',
            'MoNuSegByCancer_intraimage50pct_200x200',
        ]
    return folder_list


if __name__ == '__main__':
    # dataset = 'glysac'
    dataset = 'monuseg'

    metric_order = ['Dice', 'IoU', 'F1', 'AJI']
    model_order = ['UNet', 'nnUNet', 'MedT', 'LACSS', 'PSM']

    result_csv = 'results_%s.csv' % dataset
    result_df = pd.read_csv(result_csv)
    folder_list = get_folder_list(dataset=dataset)

    for folder in folder_list:
        summary_df = None
        result_folder = result_df[result_df['folder'] == folder].copy()

        if 'intraimage' not in folder:
            # train/test split results.

            mean_df = result_folder.groupby(['subset', 'model'])[['Dice', 'IoU', 'F1', 'AJI']].mean().reset_index()
            std_df = result_folder.groupby(['subset', 'model'])[['Dice', 'IoU', 'F1', 'AJI']].std().reset_index().fillna(0)
            summary_df = mean_df.copy()
            for metric in ['Dice', 'IoU', 'F1', 'AJI']:
                summary_df[metric] = \
                    mean_df[metric].map('${:,.3f}$'.format).astype(str) + '{\scriptsize $\color{gray}{\pm ' + std_df[metric].map('{:,.3f}'.format).astype(str) + '}$} &'

            melted_df = summary_df.melt(id_vars=['subset', 'model'], var_name='metric', value_name='value')
            melted_df['model'] = pd.Categorical(melted_df['model'], categories=model_order, ordered=True)
            melted_df['metric'] = pd.Categorical(melted_df['metric'], categories=metric_order, ordered=True)
            summary_df = melted_df.pivot(index=['subset', 'metric'], columns='model', values='value').reset_index()

        else:
            # intra-image generalization results.

            # First, Correct naming. A 'subset' here is actually a 'sample'.
            result_folder['sample'] = result_folder['subset']
            result_folder['subset'] = result_folder['sample'].map(lambda x: x.split('_img')[0])
            # Group by subset again.
            mean_df = result_folder.groupby(['subset', 'model'])[['Dice', 'IoU', 'F1', 'AJI']].mean().reset_index()
            std_df = result_folder.groupby(['subset', 'model'])[['Dice', 'IoU', 'F1', 'AJI']].std().reset_index().fillna(0)
            summary_df = mean_df.copy()
            for metric in ['Dice', 'IoU', 'F1', 'AJI']:
                summary_df[metric] = \
                    mean_df[metric].map('${:,.3f}$'.format).astype(str) + '{\scriptsize $\color{gray}{\pm ' + std_df[metric].map('{:,.3f}'.format).astype(str) + '}$} &'

            melted_df = summary_df.melt(id_vars=['subset', 'model'], var_name='metric', value_name='value')
            melted_df['model'] = pd.Categorical(melted_df['model'], categories=model_order, ordered=True)
            melted_df['metric'] = pd.Categorical(melted_df['metric'], categories=metric_order, ordered=True)
            summary_df = melted_df.pivot(index=['subset', 'metric'], columns='model', values='value').reset_index()

        print('\n\nFolder: ', folder)
        print(summary_df.to_string())

