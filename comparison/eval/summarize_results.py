import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def get_folder_list(dataset: str):
    if dataset == 'glysac':
        folder_list = [
            'GLySACByTumor_200x200',
        ]
    elif dataset == 'monuseg':
        folder_list = [
            'MoNuSegByCancer_200x200',
        ]
    return folder_list


def barplot(mean_df, std_df):
    plt.rcParams["font.family"] = 'serif'
    palette = ['#f3c80d', '#e59f05', '#d26101',
               '#a2d94d', '#009e74',
               '#72d3d4', '#57b4e9', '#2c94cd', '#0073b1',
               '#8f69c5']
    model_list = ['UNet', 'nnUNet', 'MedT', 'PSM', 'LACSS', 'SAM', 'SAM2', 'SAM_Med2D', 'MedSAM', 'Ours_gt_loc']
    model_name_list = ['UNet (Supervised)', 'nnUNet (Supervised)', 'MedT (Supervised)',
                       'PSM', 'LACSS', 'SAM', 'SAM2', 'SAM_Med2D', 'MedSAM', 'DiffKillR (ours)']

    fig = plt.figure(figsize=(12, 6))
    for subset_idx, subset in enumerate(['Breast', 'Colon', 'Prostate']):
        for metric_idx, metric in enumerate(['Dice', 'AJI']):
            ax = fig.add_subplot(2, 3, 3*metric_idx + subset_idx + 1)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='both', which='major', labelsize=12)

            subset_mean_df = mean_df[mean_df['subset'] == subset][['model', metric]]
            subset_mean_df = subset_mean_df[subset_mean_df['model'].isin(model_list)]
            subset_mean_df['model'] = pd.Categorical(subset_mean_df['model'], categories=model_list, ordered=True)
            subset_mean_df = subset_mean_df.sort_values('model')

            subset_std_df = std_df[mean_df['subset'] == subset][['model', metric]]
            subset_std_df = subset_std_df[subset_std_df['model'].isin(model_list)]
            subset_std_df['model'] = pd.Categorical(subset_std_df['model'], categories=model_list, ordered=True)
            subset_std_df = subset_std_df.sort_values('model')

            bars = ax.bar(np.arange(len(model_list)),
                   subset_mean_df[metric],
                   yerr=subset_std_df[metric],
                   edgecolor=(0, 0, 0, 1),
                   color=palette)

            ax.set_xticks([])
            ax.set_xlabel(subset + ' Cancer', fontsize=15)

            if subset_idx == 0 and metric_idx == 0:
                ax.legend(column_to_row_major(bars), column_to_row_major(model_name_list), ncol=5, bbox_to_anchor=(0.17, 1.05))

            if subset_idx == 0:
                ax.set_ylabel(metric, fontsize=16)

    fig.tight_layout(pad=2)
    fig.savefig('barplot.png', dpi=200)
    return

def column_to_row_major(lst, rows=5, cols=2):
    """
    Reorder a list from column-major to row-major order.

    Parameters:
    - lst (list): The input list in column-major order.
    - rows (int): Number of rows in the grid.
    - cols (int): Number of columns in the grid.

    Returns:
    - list: The reordered list in row-major order.
    """
    # Initialize an empty list to store row-major ordered elements
    row_major = [None] * (rows * cols)

    # Convert from column-major to row-major order
    for r in range(rows):
        for c in range(cols):
            row_major[r * cols + c] = lst[c * rows + r]

    return row_major

if __name__ == '__main__':
    # dataset = 'glysac'
    dataset = 'monuseg'

    metric_order = ['Dice', 'IoU', 'F1', 'AJI']
    model_order = ['UNet', 'nnUNet', 'MedT', 'PSM', 'LACSS', 'SAM', 'SAM2', 'SAM_Med2D', 'MedSAM', 'Ours', 'Ours_gt_loc']

    result_csv = 'results_%s.csv' % dataset
    result_df = pd.read_csv(result_csv)
    folder_list = get_folder_list(dataset=dataset)

    for folder in folder_list:
        summary_df_latex = None
        result_folder = result_df[result_df['folder'] == folder].copy()

        if 'intraimage' not in folder:
            # train/test split results.

            mean_df = result_folder.groupby(['subset', 'model'])[['Dice', 'IoU', 'F1', 'AJI']].mean().reset_index()
            std_df = result_folder.groupby(['subset', 'model'])[['Dice', 'IoU', 'F1', 'AJI']].std().reset_index().fillna(0)
            summary_df_latex = mean_df.copy()
            for metric in ['Dice', 'IoU', 'F1', 'AJI']:
                summary_df_latex[metric] = \
                    mean_df[metric].map('${:,.3f}$'.format).astype(str) + '{\scriptsize $\color{gray}{\pm ' + std_df[metric].map('{:,.3f}'.format).astype(str) + '}$} &'

            melted_df = summary_df_latex.melt(id_vars=['subset', 'model'], var_name='metric', value_name='value')
            melted_df['model'] = pd.Categorical(melted_df['model'], categories=model_order, ordered=True)
            melted_df['metric'] = pd.Categorical(melted_df['metric'], categories=metric_order, ordered=True)
            summary_df_latex = melted_df.pivot(index=['subset', 'metric'], columns='model', values='value').reset_index()

        else:
            # intra-image generalization results.

            # First, Correct naming. A 'subset' here is actually a 'sample'.
            result_folder['sample'] = result_folder['subset']
            result_folder['subset'] = result_folder['sample'].map(lambda x: x.split('_img')[0])
            # Group by subset again.
            mean_df = result_folder.groupby(['subset', 'model'])[['Dice', 'IoU', 'F1', 'AJI']].mean().reset_index()
            std_df = result_folder.groupby(['subset', 'model'])[['Dice', 'IoU', 'F1', 'AJI']].std().reset_index().fillna(0)
            summary_df_latex = mean_df.copy()
            for metric in ['Dice', 'IoU', 'F1', 'AJI']:
                summary_df_latex[metric] = \
                    mean_df[metric].map('${:,.3f}$'.format).astype(str) + '{\scriptsize $\color{gray}{\pm ' + std_df[metric].map('{:,.3f}'.format).astype(str) + '}$} &'

            melted_df = summary_df_latex.melt(id_vars=['subset', 'model'], var_name='metric', value_name='value')
            melted_df['model'] = pd.Categorical(melted_df['model'], categories=model_order, ordered=True)
            melted_df['metric'] = pd.Categorical(melted_df['metric'], categories=metric_order, ordered=True)
            summary_df_latex = melted_df.pivot(index=['subset', 'metric'], columns='model', values='value').reset_index()

        print('\n\nFolder: ', folder)
        print(summary_df_latex.to_string())

        barplot(mean_df, std_df)