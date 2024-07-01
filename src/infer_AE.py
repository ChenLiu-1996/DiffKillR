
import os
from typing import Tuple
from glob import glob
import torch
import cv2
import numpy as np
import argparse
import yaml

import phate
import scprep
import matplotlib.pyplot as plt

from utils.attribute_hashmap import AttributeHashmap
from utils.metrics import clustering_accuracy, topk_accuracy, embedding_mAP
from utils.log_util import log
from utils.parse import parse_settings
from utils.seed import seed_everything
from model.autoencoder import AutoEncoder
from datasets.BCCD_augmented import BCCD_augmentedDataset

def label_to_idx(labels):
    ''' Convert labels to indices.'''
    unique_labels = np.unique(labels)
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for idx, label in enumerate(unique_labels)}

    return label_to_idx, idx_to_label

def idx_to_label(indices, idx_to_label):
    ''' Convert indices to labels.'''
    return [idx_to_label[idx] for idx in indices]

def load_image(path: str, target_dim: Tuple[int] = None) -> np.array:
    ''' Load image as numpy array from a path string.'''
    if target_dim is not None:
        image = np.array(
            cv2.resize(
                cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR),
                             code=cv2.COLOR_BGR2RGB), target_dim))
    else:
        image = np.array(
            cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR),
                         code=cv2.COLOR_BGR2RGB))

    # Normalize image.
    image = (image / 255 * 2) - 1

    # Channel last to channel first to comply with Torch.
    image = np.moveaxis(image, -1, 0)

    return image


def load_cell_bank(folders, orig_only=True, celltype_pos=2, has_masks=False):
    '''
        Load cell bank images and labels.
        Args:
            folders: List of folders containing cell images.
            orig_only: If True, only load original images.
            celltype_pos: Position of cell type in the file name.
            has_masks: If True, load masks as well.
    '''
    cell_files = []
    cell_basenames = []

    # Load base names of cell files.
    for folder in folders:
        files = glob(os.path.join(folder, '*.png'))
        basenames = [os.path.basename(f) for f in files]
        if orig_only:
            basenames = [f for f in basenames if 'original' in f]
        cell_basenames.extend(basenames)
    
    # Deduplicate cell basenames
    # TODO: We can avoid this step by saving the original files in a separate folder.
    print(f'Before dedupe: {len(cell_basenames)}')
    cell_basenames = list(set(cell_basenames))
    print(f'After dedupe: {len(cell_basenames)}')

    # Load cell files.
    orig_loaded = []
    for folder in folders:
        for cell_basename in cell_basenames:
            if 'original' in cell_basename:
                if cell_basename in orig_loaded:
                    continue
                else:
                    orig_loaded.append(cell_basename)

            cell_file = os.path.join(folder, cell_basename)
            cell_files.append(cell_file)

    cell_labels = [os.path.basename(cell_file).split('_')[celltype_pos] for cell_file in cell_files]
    cell_images = [load_image(cell_file, target_dim=(32, 32)) for cell_file in cell_files]

    cell_images = np.array(cell_images)
    cell_labels = np.array(cell_labels)

    print(f'Done loading cell bank. Images: {cell_images.shape}, Labels: {cell_labels.shape}')

    return cell_images, cell_labels, cell_files



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference Entry point.')
    parser.add_argument('--mode', default='infer')
    parser.add_argument('--gpu-id', help='Index of GPU device', default=0)
    parser.add_argument('--run_count', help='Provide this during testing!', default=1)
    parser.add_argument('--config',
                        help='Path to config yaml file.',
                        required=True)
    parser.add_argument('--num-workers', help='Number of workers, e.g. use number of cores', default=4, type=int)
    args = vars(parser.parse_args())

    args = AttributeHashmap(args)
    config = AttributeHashmap(yaml.safe_load(open(args.config)))
    config.config_file_name = args.config
    config.gpu_id = args.gpu_id
    config.num_workers = args.num_workers
    config = parse_settings(config, log_settings=args.mode == 'train', run_count=args.run_count)

    seed_everything(config.random_seed)

    test_files = glob(config.test_img_folder + '*.png')
    print('Test files:', len(test_files))
    test_labels = [os.path.basename(test_file).split('_')[2] for test_file in test_files]
    label2idx, idx2label = label_to_idx(test_labels)
    test_labels = np.array([label2idx[label] for label in test_labels])
    test_images = [load_image(test_file, target_dim=config.target_dim) for test_file in test_files]
    print(test_images[0].shape)
    test_images = np.array(test_images)

    test_images = torch.tensor(test_images, dtype=torch.float32)
    print('Test images:', test_images.shape)
    test_loader = torch.utils.data.DataLoader(test_images, batch_size=32, shuffle=False)

    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')
    # check apple metal 
    # if torch.backends.mps.is_available():
    #     device = "mps"

    # Build the model
    try:
        model = globals()[config.model](num_filters=config.num_filters,
                                        in_channels=3,
                                        out_channels=3)
    except:
        raise ValueError('`config.model`: %s not supported.' % config.model)

    model = model.to(device)
    model.load_weights(config.model_save_path, device=device)
    log('%s: Model weights successfully loaded.' % config.model,
        to_console=True)
    
    model.eval()
    test_embeddings = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            recons, latent = model(data)
            latent = torch.flatten(latent, start_dim=1)
            test_embeddings.append(latent.cpu().numpy())
    
    test_embeddings = np.concatenate(test_embeddings, axis=0)
    print('Done infering test embeddings:', test_embeddings.shape)

    # Load reference embeddings from cell bank.
    aug_methods = config.aug_methods.split(',')
    config.cell_bank_folders = [f'{config.dataset_path}/{aug_method}/image/' for aug_method in aug_methods]
    print('Cell bank folders:', config.cell_bank_folders)
    reference_images, reference_labels, _ = load_cell_bank(config.cell_bank_folders, 
                                                        orig_only=True)
    reference_labels = np.array([label2idx[label] for label in reference_labels])
    reference_images = torch.tensor(reference_images, dtype=torch.float32)
    reference_loader = torch.utils.data.DataLoader(reference_images, batch_size=32, shuffle=False)
    reference_embeddings = []
    with torch.no_grad():
        for i, data in enumerate(reference_loader):
            data = data.to(device)
            recons, latent = model(data)
            latent = torch.flatten(latent, start_dim=1)
            reference_embeddings.append(latent.cpu().numpy())
    reference_embeddings = np.concatenate(reference_embeddings, axis=0)
    print('Done infering reference embeddings:', reference_embeddings.shape)

    # Match test with reference embeddings.
    print('test labels:', test_labels.shape, test_labels[:10])
    print('reference labels:', reference_labels.shape, reference_labels[:10])
    accuracy = clustering_accuracy(test_embeddings, reference_embeddings, 
                                    test_labels, reference_labels)
    

    print('Clustering accuracy:', accuracy)

    # Visulize the embeddings
    plt.rcParams['font.family'] = 'serif'
    fig_embedding = plt.figure(figsize=(10, 8 * 2))

    # Embedding
    all_embeddings = np.concatenate([test_embeddings, reference_embeddings], axis=0)
    phate_op = phate.PHATE()
    phate_embedding = phate_op.fit_transform(all_embeddings)

    ax_embedding = fig_embedding.add_subplot(2, 1, 1)
    conditions = np.concatenate([np.ones(test_labels.shape), np.zeros(reference_labels.shape)])
    # Map legend to labels
    conditions = idx_to_label(conditions, {0: 'Reference', 1: 'Test'})
    scprep.plot.scatter2d(phate_embedding, c=conditions, 
                        label_prefix='PHATE', ticks=False, legend=True, ax=ax_embedding)
    
    ax_embedding.set_title('PHATE Embedding')

    ax_embedding = fig_embedding.add_subplot(2, 1, 2)
    colors = np.concatenate([test_labels, reference_labels])
    colors = idx_to_label(colors, idx2label)
    scprep.plot.scatter2d(phate_embedding, c=colors,
                        label_prefix='PHATE', ticks=False, legend=True, ax=ax_embedding)
    ax_embedding.set_title(f'Accuracy: {accuracy:.3f}')
    
    # save the plot
    plt.savefig(f'{config.output_save_path}/PHATE_embedding.png')

    


