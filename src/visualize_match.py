import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml

from infer_AE import load_image
from utils.attribute_hashmap import AttributeHashmap
from utils.parse import parse_settings


def visualize_match(df: pd.DataFrame, save_dir: str, n: int = 5, ):
    '''
        Visualize the matching between test and reference images.
    '''
    n_rows = n
    n_cols = 2
    
    fig = plt.figure(figsize=(2, 2 * n_rows))
    # set font
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.size'] = 10

    sampled_idxs = np.random.choice(df.index, n_rows, replace=False)

    for i, idx in enumerate(sampled_idxs):
        test_img = load_image(df.loc[idx, 'test_file'])
        ref_img = load_image(df.loc[idx, 'matching_reference_file'])

        test_img = np.moveaxis(test_img, 0, -1)
        ref_img = np.moveaxis(ref_img, 0, -1)

        ax = fig.add_subplot(n_rows, n_cols, 2 * i + 1)
        ax.imshow(test_img)
        ax.set_title(f'Test')
        ax.axis('off')

        ax = fig.add_subplot(n_rows, n_cols, 2 * i + 2)
        ax.imshow(ref_img)
        ax.set_title(f'Ref: {df.loc[idx, "matching_celltype"]}')
        ax.axis('off')
    

    plt.tight_layout()
    # plt.show()

    save_path = os.path.join(save_dir, 'match_visualization.png')
    plt.savefig(save_path)

    print(f'Saved the match visualization to {save_path}')


if __name__ == '__main__':
    config_path = './config/ours_config.yaml'
    config = AttributeHashmap(yaml.safe_load(open(config_path)))
    config = parse_settings(config, log_settings=False, run_count=1)

    print('Config:', config)

    df = pd.read_csv(config.matched_pair_path)
    save_dir = config.output_save_path

    visualize_match(df, save_dir, n=10)
    
