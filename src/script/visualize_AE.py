import argparse
import numpy as np
import torch
import os
import sys
import cv2
import yaml
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir + '/')
from model.autoencoder import AutoEncoder
from utils.attribute_hashmap import AttributeHashmap
from utils.prepare_dataset import prepare_dataset
from utils.log_util import log
from utils.parse import parse_settings
from utils.seed import seed_everything
from datasets.augmented import load_image


def construct_triplet_batch(img_paths,
                            latent_features,
                            num_pos,
                            num_neg,
                            model,
                            dataset,
                            split,
                            device):
    '''
        Returns:
        pos_features: (bsz * num_pos, latent_dim)
        neg_features: (bsz * num_neg, latent_dim)

    '''
    pos_images = None
    cell_type_labels = [] # (bsz)
    pos_cell_type_labels = [] # (bsz * num_pos)

    # Positive.
    for img_path in img_paths:
        cell_type = dataset.get_celltype(img_path=img_path)
        cell_type_labels.append(dataset.cell_type_to_idx[cell_type])
        pos_cell_type_labels.extend([dataset.cell_type_to_idx[cell_type]] * num_pos)

        aug_images, _ = dataset.sample_celltype(split=split,
                                                celltype=cell_type,
                                                cnt=num_pos)
        aug_images = torch.Tensor(aug_images).to(device)

        if pos_images is not None:
            pos_images = torch.cat([pos_images, aug_images], dim=0)
        else:
            pos_images = aug_images
    _, pos_features = model(pos_images) # (bsz * num_pos, latent_dim)

    # Negative.
    num_neg = config.num_neg
    neg_features = None # (bsz*num_neg, latent_dim)
    all_features = torch.cat([latent_features, pos_features], dim=0) # (bsz * (1+num_pos), latent_dim)

    all_cell_type_labels = cell_type_labels.copy()
    all_cell_type_labels.extend(pos_cell_type_labels) # (bsz * (1+num_pos))

    for img_path in img_paths:
        cell_type = dataset.get_celltype(img_path=img_path)

        negative_pool = np.argwhere(
            (np.array(all_cell_type_labels) != dataset.cell_type_to_idx[cell_type]) * 1).flatten()

        neg_idxs = np.random.choice(negative_pool, size=num_neg, replace=False)

        if neg_features is not None:
            neg_features = torch.cat([neg_features, all_features[neg_idxs]], dim=0)
        else:
            neg_features = all_features[neg_idxs]

    return pos_features, neg_features



def construct_batch_images_with_n_views(images, img_paths, dataset, n_views, split, device):
    '''
        Returns:
        batch_images: [bsz * n_views, ...],
        cell_type_labels: [bsz * n_views]
    '''
    # Construct batch_images [bsz * n_views, in_chan, H, W].
    batch_images = None
    cell_type_labels = []
    n_views = config.n_views
    for image, img_path in zip(images, img_paths):
        cell_type = dataset.get_celltype(img_path=img_path)
        cell_type_labels.append(dataset.cell_type_to_idx[cell_type])
        if n_views > 1:
            aug_images, _ = dataset.sample_celltype(split=split,
                                                    celltype=cell_type,
                                                    cnt=n_views-1)
            aug_images = torch.Tensor(aug_images).to(device) # (cnt, in_chan, H, W)

            image = torch.unsqueeze(image, dim=0) # (1, in_chan, H, W)
            if batch_images is not None:
                batch_images = torch.cat([batch_images, image, aug_images], dim=0)
            else:
                batch_images = torch.cat([image, aug_images], dim=0)
        else:
            if batch_images is not None:
                batch_images = torch.cat([batch_images, image], dim=0)
            else:
                batch_images = torch.cat([image], dim=0)

    batch_images = batch_images.float().to(device) #[bsz * n_views, in_chan, H, W].

    return batch_images, cell_type_labels


class InferenceImagePatches(Dataset):
    def __init__(self,
                 image_path: str,
                 target_dim: Tuple[int] = (96, 96),
                 num_patches_h: int = 100,
                 num_patches_w: int = 100):

        super().__init__()

        self.target_dim = target_dim
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w

        raw_image = load_image(image_path)
        assert len(raw_image.shape) == 3
        assert raw_image.shape[0] == 3

        _, raw_image_H, raw_image_W = raw_image.shape
        assert raw_image_H > self.num_patches_h * target_dim[0]
        assert raw_image_W > self.num_patches_w * target_dim[1]
        delta_h = self.num_patches_h * target_dim[0]//2
        delta_w = self.num_patches_w * target_dim[1]//2

        image = raw_image[:,
            raw_image_H//2 - delta_h : raw_image_H//2 + delta_h,
            raw_image_W//2 - delta_w : raw_image_W//2 + delta_w]

        self.image_list = []
        for i in range(self.num_patches_h):
            for j in range(self.num_patches_w):
                self.image_list.append(image[:,
                                             i*target_dim[0]:(i+1)*target_dim[0],
                                             j*target_dim[1]:(j+1)*target_dim[1]])
        self.image_list = np.array(self.image_list)

    def __len__(self) -> int:
        return self.num_patches_h * self.num_patches_w

    def __getitem__(self, idx) -> np.array:
        return self.image_list[idx]


@torch.no_grad()
def test(config: AttributeHashmap, image_path: str):
    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')
    dataset, train_set, val_set, test_set = prepare_dataset(config=config)

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

    save_path_fig_embeddings_tvt = '%s/results/embeddings_wrt_train_val_test.png' % config.output_save_path
    save_path_fig_embeddings_inf = '%s/results/embeddings_wrt_inference.png' % config.output_save_path
    os.makedirs(os.path.dirname(save_path_fig_embeddings_tvt), exist_ok=True)

    # Test.
    model.eval()

    # Visualize latent embeddings.
    embeddings = {split: None for split in ['train', 'val', 'test']}
    embedding_labels = {split: [] for split in ['train', 'val', 'test']}
    og_inputs = {split: None for split in ['train', 'val', 'test']}
    reconstructed = {split: None for split in ['train', 'val', 'test']}

    inference_image_patches = InferenceImagePatches(image_path=image_path, target_dim=config.target_dim)
    inference_image_loader = DataLoader(dataset=inference_image_patches,
                                        batch_size=config.batch_size,
                                        shuffle=False,
                                        num_workers=config.num_workers)

    with torch.no_grad():
        embeddings_inference = None
        for images in tqdm(inference_image_loader):
            images = images.float().to(device)
            _, latent_features = model(images)
            latent_features = torch.flatten(latent_features, start_dim=1)
            latent_features = latent_features.cpu()
            if embeddings_inference is None:
                embeddings_inference = latent_features  # (bsz, latent_dim)
            else:
                embeddings_inference = torch.cat([embeddings_inference, latent_features], dim=0)
        embeddings_inference = embeddings_inference.numpy()

        for split, split_set in zip(['train', 'val', 'test'], [train_set, val_set, test_set]):
            for iter_idx, (images, _, canonical_images, _, img_paths) in enumerate(tqdm(split_set)):
                images = images.float().to(device)
                recon_images, latent_features = model(images)
                latent_features = torch.flatten(latent_features, start_dim=1)

                # Move to cpu to save memory on gpu.
                images = images.cpu()
                recon_images = recon_images.cpu()
                latent_features = latent_features.cpu()

                if embeddings[split] is None:
                    embeddings[split] = latent_features  # (bsz, latent_dim)
                else:
                    embeddings[split] = torch.cat([embeddings[split], latent_features], dim=0)
                if reconstructed[split] is None:
                    reconstructed[split] = recon_images # (bsz, in_chan, H, W)
                else:
                    reconstructed[split] = torch.cat([reconstructed[split], recon_images], dim=0)
                if og_inputs[split] is None:
                    og_inputs[split] = images
                else:
                    og_inputs[split] = torch.cat([og_inputs[split], images], dim=0)
                embedding_labels[split].extend([dataset.get_celltype(img_path=img_path) for img_path in img_paths])

            embeddings[split] = embeddings[split].numpy()
            reconstructed[split] = reconstructed[split].numpy()
            og_inputs[split] = og_inputs[split].numpy()
            embedding_labels[split] = np.array(embedding_labels[split])
            assert len(embeddings[split]) == len(reconstructed[split]) == len(embedding_labels[split])

    # Plot latent embeddings
    import phate
    import scprep
    import matplotlib.pyplot as plt

    plt.rcParams['font.family'] = 'serif'

    # Use PHATE on the train/val/test data.
    fig_embedding = plt.figure(figsize=(10, 6 * 3))
    for split in ['train', 'val', 'test']:
        phate_op = phate.PHATE(random_state=0,
                               n_jobs=1,
                               n_components=2,
                               verbose=False)
        data_phate = phate_op.fit_transform(embeddings[split])
        data_phate_inference = phate_op.transform(embeddings_inference)

        print('Visualizing ', split, ' : ',  data_phate.shape)
        ax = fig_embedding.add_subplot(3, 1, ['train', 'val', 'test'].index(split) + 1)
        scprep.plot.scatter2d(data_phate_inference,
                              c='grey',
                              legend=None,
                              ax=ax,
                              title=split,
                              xticks=False,
                              yticks=False,
                              label_prefix='PHATE',
                              fontsize=10,
                              s=5,
                              alpha=0.5)
        scprep.plot.scatter2d(data_phate,
                              c=embedding_labels[split],
                              legend=dataset.cell_types,
                              ax=ax,
                              title=split,
                              xticks=False,
                              yticks=False,
                              label_prefix='PHATE',
                              fontsize=10,
                              s=5,
                              alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path_fig_embeddings_tvt)
    plt.close(fig_embedding)

    # Use PHATE on the inference data.
    fig_embedding = plt.figure(figsize=(10, 6 * 3))
    phate_op = phate.PHATE(random_state=0,
                           n_jobs=1,
                           n_components=2,
                           verbose=False)
    data_phate_inference = phate_op.fit_transform(embeddings_inference)

    for split in ['train', 'val', 'test']:
        data_phate = phate_op.transform(embeddings[split])

        print('Visualizing ', split, ' : ',  data_phate.shape)
        ax = fig_embedding.add_subplot(3, 1, ['train', 'val', 'test'].index(split) + 1)
        scprep.plot.scatter2d(data_phate_inference,
                              c='grey',
                              legend=None,
                              ax=ax,
                              title=split,
                              xticks=False,
                              yticks=False,
                              label_prefix='PHATE',
                              fontsize=10,
                              s=5,
                              alpha=0.5)
        scprep.plot.scatter2d(data_phate,
                              c=embedding_labels[split],
                              legend=dataset.cell_types,
                              ax=ax,
                              title=split,
                              xticks=False,
                              yticks=False,
                              label_prefix='PHATE',
                              fontsize=10,
                              s=5,
                              alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path_fig_embeddings_inf)
    plt.close(fig_embedding)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entry point.')
    parser.add_argument('--gpu-id', help='Index of GPU device', default=0)
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
    config = parse_settings(config, log_settings=False)

    seed_everything(config.random_seed)

    raw_image_path = '../../raw_data/A28-87_CP_lvl1_HandE_1_Merged_RAW_ch00.tif'

    test(config=config, image_path=raw_image_path)
