import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    """Triplet loss function.
    It also supports multi-positive, multi-negative triplets."""
    def __init__(self, distance_measure='cosine', margin=1.0, num_pos=1, num_neg=1, pnorm=2):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.num_pos = num_pos
        self.num_neg = num_neg
        self.pnorm = pnorm
        self.distance_measure = distance_measure

        assert num_neg == num_pos, 'num_neg must equal num_pos for current implementation'
        assert distance_measure in ('cosine', 'norm')

    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: anchor embeddings, tensor of shape [bsz, ...]
            positive: positive embeddings, tensor of shape [bsz, num_pos, ...]
            negative: negative embeddings, tensor of shape [bsz, num_neg, ...]
            NOTE: positive[b, i, ...] is the i-th positive embedding of the b-th anchor
            # TODO: normalize + consine similarity + margin -> 1.0
        Returns:
            triplet_loss: scalar loss
        """
        assert self.num_pos == self.num_neg, 'num_pos must equal num_neg for current implementation'
        bsz = anchor.shape[0]
        if len(anchor.shape) > 2:
            anchor = anchor.view(anchor.shape[0], -1) # (bsz, latent_dim)
        if len(positive.shape) > 3:
            positive = positive.view(positive.shape[0], positive.shape[1], -1)
        if len(negative.shape) > 3:
            negative = negative.view(negative.shape[0], negative.shape[1], -1)

        # Normalize the feature vectors
        if self.distance_measure == 'cosine':
            anchor = F.normalize(anchor, p=2, dim=1)
            positive = F.normalize(positive, p=2, dim=2)
            negative = F.normalize(negative, p=2, dim=2)
        
        # Tile anchor to shape (bsz * num_pos, latent_dim)
        anchor = anchor.repeat(self.num_pos, 1) # (bsz * num_pos, latent_dim)

        # Reshape positive and negative to shape (bsz * num_pos, latent_dim)
        positive = torch.cat(torch.unbind(positive, dim=1), dim=0)
        negative = torch.cat(torch.unbind(negative, dim=1), dim=0)
        
        # Pairwise distances
        if self.distance_measure == 'cosine':
            d_ap = 1 - F.cosine_similarity(anchor, positive, dim=1)
            d_an = 1 - F.cosine_similarity(anchor, negative, dim=1)
        elif self.distance_measure == 'norm':
            d_ap = F.pairwise_distance(anchor, positive, p=self.pnorm) # (bsz * num_pos)
            d_an = F.pairwise_distance(anchor, negative, p=self.pnorm) # (bsz * num_neg)
        else:
            raise ValueError('Invalid distance measure: %s' % self.distance_measure)

        # Compute loss
        losses = F.relu(d_ap - d_an + self.margin) # (bsz * num_pos)

        # Mean over positive pairs, and batch
        triplet_loss = losses.sum() / self.num_pos / bsz

        return triplet_loss

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