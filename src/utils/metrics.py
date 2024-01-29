import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from skimage.metrics import structural_similarity

def clustering_accuracy(embeddings: np.ndarray,
                        reference_embeddings: np.ndarray,
                        labels: np.ndarray,
                        reference_labels: np.ndarray,
                        distance_measure = 'cosine',
                        voting_k:int = 1) -> float:
    '''
    Compute clustering accuracy for a given set of embeddings, references and labels.

    Args:
        embeddings: [N1, D] array of embeddings.
        reference_embeddings: [N2, D] array of reference embeddings.
        labels: [N1,] array of labels.
        reference_labels: [N2, ] array of reference labels.
        distance_measure: str, either 'cosine' or 'norm'.
        voting_k: int, how many nearest neighbors to vote for the label.
    '''
    assert embeddings.shape[0] == labels.shape[0], \
        'Embeddings and labels must have the same number of nodes.'
    assert reference_embeddings.shape[0] == reference_labels.shape[0], \
        'Embeddings and labels must have the same number of nodes.'
    assert distance_measure in ['cosine', 'norm'], \
        'Invalid distance measure: %s' % distance_measure

    N1, N2 = embeddings.shape[0], reference_embeddings.shape[0]
    if voting_k > N2:
        voting_k = N2

    # Compute pairwise distances, shape [N1, N2]
   
    if distance_measure == 'cosine':
        distances = 1 - cosine_similarity(embeddings, reference_embeddings)
    elif distance_measure == 'norm':
        distances = euclidean_distances(embeddings, reference_embeddings)

    # Get the votingk node indices for each node
    voting_nodes = np.argsort(distances, axis=1)[:, :voting_k] # [N1, voting_k]

    # Get the votingk labels for each node
    voting_labels = reference_labels[voting_nodes].astype(int) # [N1, voting_k]
    print('voting_labels: ', voting_labels.shape)

    # Get the most frequent label for each node
    predicted_labels = np.array([np.argmax(np.bincount(voting_labels[i])) for i in range(N1)]) # [N1,]

    # Compute accuracy
    acc = np.mean((predicted_labels == labels) * 1.0)

    return acc


def topk_accuracy(embeddings: np.ndarray,
                  adj_mat: np.ndarray,
                  distance_measure='cosine',
                  k: int = 5) -> float:
    '''
    Compute top-k accuracy for a given set of embeddings and adjacency matrix.
    It computes the top-k accuracy for each node, and then average over all nodes.
    "What's percentage of the top-k nearest neighbors are my actual connected neighors?"

    Args:
        embeddings: [N, D] array of embeddings.
        adj_mat: [N, N] binary array of adjacency matrix.
        distance_measure: str, either 'cosine' or 'norm'.
        k: int or None. If None, k = np.sum(adj_mat, axis=1) for each node
    '''
    assert embeddings.shape[0] == adj_mat.shape[0], \
        'Embeddings and adjacency matrix must have the same number of nodes.'
    assert distance_measure in ['cosine', 'norm'], \
        'Invalid distance measure: %s' % distance_measure

    N = embeddings.shape[0]

    # Compute pairwise distances
    if distance_measure == 'cosine':
        distances = 1 - cosine_similarity(embeddings, embeddings)
    elif distance_measure == 'norm':
        distances = euclidean_distances(embeddings, embeddings)

    # Get the top-k node indices for each node
    topk_nodes = np.squeeze(np.argsort(distances, axis=1)[:, :k]) # [N, k]
    print('topk_nodes: ', topk_nodes.shape)

    # Get the top-k labels for each node
    topk_labels = []
    for i in range(k):
        topk_labels.append(adj_mat[np.arange(N), topk_nodes[np.arange(N), i]]) # [N,]
    topk_labels = np.hstack(topk_labels) # [N, k]
    print('topklabels: ', topk_labels.shape)

    if k > 1:
        acc = np.mean(np.sum(topk_labels, axis=1) / k)
    else:
        acc = np.mean(topk_labels)
    
    return acc


def embedding_mAP(embeddings: np.ndarray,
                  graph_adjacency: np.ndarray,
                  distance_op: str = 'norm') -> float:
    '''
    embeddings: ndarray (N, embedding_dim)
    graph_adjacency: ndarray (N, N)
    distance_op: str, 'norm'|'dot'|'cosine'

    AP_(xi, xj) := \frac{the number of neighbors of xi, \
    enclosed by smallest ball that contains xj centered at xi}{the points enclosed by the ball centered at xi}

    graph_adjacency[i, j] = 1 if i and j are neighbors, 0 otherwise

    Returns:
        mAP: float
    '''
    N = embeddings.shape[0]
    assert N == graph_adjacency.shape[0] == graph_adjacency.shape[1]

    # compute the distance matrix
    distance_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if distance_op == 'norm':
                distance_matrix[i, j] = np.linalg.norm(embeddings[i] - embeddings[j])
            elif distance_op == 'dot':
                distance_matrix[i, j] = np.dot(embeddings[i], embeddings[j])
            elif distance_op == 'cosine':
                distance_matrix[i, j] = np.dot(embeddings[i], embeddings[j]) / \
                                        (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
            else:
                raise Exception('distance_op must be either norm or dot')

    # compute the AP
    AP = np.zeros(N)
    for i in range(N):
        # find the neighbors of i
        neighbors = np.argwhere(graph_adjacency[i] == 1).flatten()
        # compute the distance between i and its neighbors
        distances = distance_matrix[i, neighbors] # (n_neighbors, )
        for j in range(len(neighbors)):
            # compute the number of points enclosed by the ball_j centered at i
            all_enclosed = np.argwhere(distance_matrix[i] <= distances[j]).flatten()
            # compute the number of neighbors of enclosed by the ball_j centered at i
            n_enclosed_j = len(np.intersect1d(all_enclosed, neighbors))
            # compute the AP
            if n_enclosed_j > 0:
                AP[i] += n_enclosed_j / all_enclosed.shape[0]

        if len(neighbors) > 0:
            AP[i] /= len(neighbors)

    mAP = np.mean(AP)

    return mAP


def psnr(image1, image2, max_value=2):
    '''
    Assuming data range is [-1, 1].
    '''
    assert image1.shape == image2.shape

    eps = 1e-12

    mse = np.mean((image1 - image2)**2)
    return 20 * np.log10(max_value / np.sqrt(mse + eps))


def ssim(image1: np.array, image2: np.array, data_range=2, **kwargs) -> float:
    '''
    Please make sure the data are provided in [H, W, C] shape.

    Assuming data range is [-1, 1] --> `data_range` = 2.
    '''
    assert image1.shape == image2.shape

    H, W = image1.shape[:2]

    if min(H, W) < 7:
        win_size = min(H, W)
        if win_size % 2 == 0:
            win_size -= 1
    else:
        win_size = None

    if len(image1.shape) == 3:
        channel_axis = -1
    else:
        channel_axis = None

    return structural_similarity(image1,
                                 image2,
                                 data_range=data_range,
                                 channel_axis=channel_axis,
                                 win_size=win_size,
                                 **kwargs)

def dice_coeff(mask1: np.array, mask2: np.array) -> float:
    '''
    Dice Coefficient between 2 binary masks.
    '''

    if isinstance(mask1.min(), bool):
        mask1 = np.uint8(mask1)
    if isinstance(mask2.min(), bool):
        mask2 = np.uint8(mask2)

    assert mask1.min() in [0, 1] and mask2.min() in [0, 1], \
        'min values for masks are not in [0, 1]: mask1: %s, mask2: %s' % (mask1.min(), mask2.min())
    assert mask1.max() == 1 and mask2.max() == 1, \
        'max values for masks are not 1: mask1: %s, mask2: %s' % (mask1.max(), mask2.max())

    assert mask1.shape == mask2.shape, \
        'mask shapes do not match: %s vs %s' % (mask1.shape, mask2.shape)

    intersection = np.logical_and(mask1, mask2).sum()
    denom = np.sum(mask1) + np.sum(mask2)
    epsilon = 1e-9

    dice = 2 * intersection / (denom + epsilon)

    return dice