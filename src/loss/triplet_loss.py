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