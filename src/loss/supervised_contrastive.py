"""
Adapted from:
    https://github.com/HobbitLong/SupContrast/blob/master/losses.py
"""
from __future__ import print_function

import torch
import torch.nn as nn

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrastive_mode='all',
                 base_temperature=0.07, normalize_features=True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrastive_mode
        self.base_temperature = base_temperature
        self.normalize_features = normalize_features

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            NOTE: n_views is the number of views per image.
            features[b, i, ...] is the i-th view of the b-th image.
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1] # n_views
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # (bsz * n_views, latent_dim)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0] # (bsz, latent_dim)
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature # (bsz * n_views, latent_dim)
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits

        # Normalize the feature vectors to avoid too large dot product values.
        if self.normalize_features == True:
            anchor_feature = nn.functional.normalize(anchor_feature, dim=1)
            contrast_feature = nn.functional.normalize(contrast_feature, dim=1)
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature) # (bsz, bsz * n_views) if one ; or (bsz * n_views, bsz * n_views) if all
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count) # (bsz, bsz * n_views) or (bsz * n_views, bsz * n_views)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask + 1e-8 # numerical stability
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss