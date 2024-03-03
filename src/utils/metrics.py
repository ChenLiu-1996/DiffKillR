import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from skimage.metrics import structural_similarity
import scipy.stats as stats
from sklearn.neighbors import NearestNeighbors
import scipy
from skimage.measure import label
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score, precision_score
from scipy.spatial.distance import directed_hausdorff as hausdorff
from scipy.ndimage.measurements import center_of_mass

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
    
    knn_op = NearestNeighbors(n_neighbors=voting_k, metric=distance_measure)
    knn_op.fit(reference_embeddings)

    _, indices = knn_op.kneighbors(embeddings, return_distance=True) # [N1, k], [N1, k]
    voting_labels = reference_labels[indices] # [N1, k]
    print('voting_labels.shape: ', voting_labels.shape)

    predicted_labels, _ = stats.mode(voting_labels, axis=1)
    print('predicted_labels.shape: ', predicted_labels.shape)

    # Compute accuracy
    acc = np.mean((np.squeeze(predicted_labels) == labels) * 1.0)

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
    topk_nodes = np.argsort(distances, axis=1)[:, :k] # [N, k]
    if len(topk_nodes.shape) < 2:
        topk_nodes = topk_nodes[:, np.newaxis]
        
    # Get the top-k labels for each node
    topk_labels = np.take_along_axis(adj_mat, topk_nodes, axis=1)

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
    distance_matrix = scipy.spatial.distance.cdist(embeddings, embeddings, metric=distance_op)

    # compute the AP
    AP = np.zeros(N)
    for i in range(N):
        # find the neighbors of i
        neighbors = np.argwhere(graph_adjacency[i] == 1).flatten()
        #print('neighbors: ', neighbors.shape, neighbors[:5])
        # compute the distance between i and its neighbors
        distances = distance_matrix[i, neighbors] # (n_neighbors, )
        #print('distances: ', distances.shape, distances[:5])
        for j in range(len(neighbors)):
            # compute the number of points enclosed by the ball_j centered at i
            all_enclosed = np.argwhere(distance_matrix[i] <= distances[j]).flatten()
            # compute the number of neighbors of enclosed by the ball_j centered at i
            n_enclosed_j = len(np.intersect1d(all_enclosed, neighbors))
            # compute the AP
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
    # assert mask1.max() == 1 and mask2.max() == 1, \
    #     'max values for masks are not 1: mask1: %s, mask2: %s' % (mask1.max(), mask2.max())

    assert mask1.shape == mask2.shape, \
        'mask shapes do not match: %s vs %s' % (mask1.shape, mask2.shape)

    intersection = np.logical_and(mask1, mask2).sum()
    denom = np.sum(mask1) + np.sum(mask2)
    epsilon = 1e-9

    dice = 2 * intersection / (denom + epsilon)

    return dice

def IoU(mask1: np.array, mask2: np.array) -> float:
    '''
    Intersection over Union between 2 binary masks.
    '''

    if isinstance(mask1.min(), bool):
        mask1 = np.uint8(mask1)
    if isinstance(mask2.min(), bool):
        mask2 = np.uint8(mask2)

    assert mask1.min() in [0, 1] and mask2.min() in [0, 1], \
        'min values for masks are not in [0, 1]: mask1: %s, mask2: %s' % (mask1.min(), mask2.min())
    # assert mask1.max() == 1 and mask2.max() == 1, \
    #     'max values for masks are not 1: mask1: %s, mask2: %s' % (mask1.max(), mask2.max())

    assert mask1.shape == mask2.shape, \
        'mask shapes do not match: %s vs %s' % (mask1.shape, mask2.shape)

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    epsilon = 1e-9

    iou = intersection / (union + epsilon)

    return iou


def compute_metrics(pred, gt, names):
    """
    Computes metrics specified by names between predicted label and groundtruth label.
    """

    gt_labeled = label(gt)
    pred_labeled = label(pred)

    gt_binary = gt_labeled.copy()
    pred_binary = pred_labeled.copy()
    gt_binary[gt_binary > 0] = 1
    pred_binary[pred_binary > 0] = 1
    gt_binary, pred_binary = gt_binary.flatten(), pred_binary.flatten()

    results = {}

    # pixel-level metrics
    if 'acc' in names:
        results['acc'] = accuracy_score(gt_binary, pred_binary)
    if 'roc' in names:
        results['roc'] = roc_auc_score(gt_binary, pred_binary)
    if 'p_F1' in names:  # pixel-level F1
        results['p_F1'] = f1_score(gt_binary, pred_binary)
    if 'p_recall' in names:  # pixel-level F1
        results['p_recall'] = recall_score(gt_binary, pred_binary)
    if 'p_precision' in names:  # pixel-level F1
        results['p_precision'] = precision_score(gt_binary, pred_binary)

    # object-level metrics
    if 'aji' in names:
        results['aji'] = AJI_fast(gt_labeled, pred_labeled)
    if 'haus' in names:
        results['dice'], results['iou'], results['haus'] = accuracy_object_level(pred_labeled, gt_labeled, True)
    elif 'dice' in names or 'iou' in names:
        results['dice'], results['iou'], _ = accuracy_object_level(pred_labeled, gt_labeled, False)

    return results


def accuracy_object_level(pred, gt, hausdorff_flag=True):
    """ Compute the object-level metrics between predicted and
    groundtruth: dice, iou, hausdorff """
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    if not isinstance(gt, np.ndarray):
        gt = np.array(gt)

    # get connected components
    pred_labeled = label(pred, connectivity=2)
    Ns = len(np.unique(pred_labeled)) - 1
    gt_labeled = label(gt, connectivity=2)
    Ng = len(np.unique(gt_labeled)) - 1

    # --- compute dice, iou, hausdorff --- #
    pred_objs_area = np.sum(pred_labeled>0)  # total area of objects in image
    gt_objs_area = np.sum(gt_labeled>0)  # total area of objects in groundtruth gt

    # compute how well groundtruth object overlaps its segmented object
    dice_g = 0.0
    iou_g = 0.0
    hausdorff_g = 0.0
    for i in range(1, Ng + 1):
        gt_i = np.where(gt_labeled == i, 1, 0)
        overlap_parts = gt_i * pred_labeled

        # get intersection objects numbers in image
        obj_no = np.unique(overlap_parts)
        obj_no = obj_no[obj_no != 0]

        gamma_i = float(np.sum(gt_i)) / gt_objs_area

        # show_figures((pred_labeled, gt_i, overlap_parts))

        if obj_no.size == 0:   # no intersection object
            dice_i = 0
            iou_i = 0

            # find nearest segmented object in hausdorff distance
            if hausdorff_flag:
                min_haus = 1e3

                # find overlap object in a window [-50, 50]
                pred_cand_indices = find_candidates(gt_i, pred_labeled)

                for j in pred_cand_indices:
                    pred_j = np.where(pred_labeled == j, 1, 0)
                    seg_ind = np.argwhere(pred_j)
                    gt_ind = np.argwhere(gt_i)
                    haus_tmp = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])

                    if haus_tmp < min_haus:
                        min_haus = haus_tmp
                haus_i = min_haus
        else:
            # find max overlap object
            obj_areas = [np.sum(overlap_parts == k) for k in obj_no]
            seg_obj = obj_no[np.argmax(obj_areas)]  # segmented object number
            pred_i = np.where(pred_labeled == seg_obj, 1, 0)  # segmented object

            overlap_area = np.max(obj_areas)  # overlap area

            dice_i = 2 * float(overlap_area) / (np.sum(pred_i) + np.sum(gt_i))
            iou_i = float(overlap_area) / (np.sum(pred_i) + np.sum(gt_i) - overlap_area)

            # compute hausdorff distance
            if hausdorff_flag:
                seg_ind = np.argwhere(pred_i)
                gt_ind = np.argwhere(gt_i)
                haus_i = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])

        dice_g += gamma_i * dice_i
        iou_g += gamma_i * iou_i
        if hausdorff_flag:
            hausdorff_g += gamma_i * haus_i

    # compute how well segmented object overlaps its groundtruth object
    dice_s = 0.0
    iou_s = 0.0
    hausdorff_s = 0.0
    for j in range(1, Ns + 1):
        pred_j = np.where(pred_labeled == j, 1, 0)
        overlap_parts = pred_j * gt_labeled

        # get intersection objects number in gt
        obj_no = np.unique(overlap_parts)
        obj_no = obj_no[obj_no != 0]

        # show_figures((pred_j, gt_labeled, overlap_parts))

        sigma_j = float(np.sum(pred_j)) / pred_objs_area
        # no intersection object
        if obj_no.size == 0:
            dice_j = 0
            iou_j = 0

            # find nearest groundtruth object in hausdorff distance
            if hausdorff_flag:
                min_haus = 1e3

                # find overlap object in a window [-50, 50]
                gt_cand_indices = find_candidates(pred_j, gt_labeled)

                for i in gt_cand_indices:
                    gt_i = np.where(gt_labeled == i, 1, 0)
                    seg_ind = np.argwhere(pred_j)
                    gt_ind = np.argwhere(gt_i)
                    haus_tmp = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])

                    if haus_tmp < min_haus:
                        min_haus = haus_tmp
                haus_j = min_haus
        else:
            # find max overlap gt
            gt_areas = [np.sum(overlap_parts == k) for k in obj_no]
            gt_obj = obj_no[np.argmax(gt_areas)]  # groundtruth object number
            gt_j = np.where(gt_labeled == gt_obj, 1, 0)  # groundtruth object

            overlap_area = np.max(gt_areas)  # overlap area

            dice_j = 2 * float(overlap_area) / (np.sum(pred_j) + np.sum(gt_j))
            iou_j = float(overlap_area) / (np.sum(pred_j) + np.sum(gt_j) - overlap_area)

            # compute hausdorff distance
            if hausdorff_flag:
                seg_ind = np.argwhere(pred_j)
                gt_ind = np.argwhere(gt_j)
                haus_j = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])

        dice_s += sigma_j * dice_j
        iou_s += sigma_j * iou_j
        if hausdorff_flag:
            hausdorff_s += sigma_j * haus_j

    return (dice_g + dice_s) / 2, (iou_g + iou_s) / 2, (hausdorff_g + hausdorff_s) / 2


def find_candidates(obj_i, objects_labeled, radius=50):
    """
    find object indices in objects_labeled in a window centered at obj_i
    when computing object-level hausdorff distance
    """
    if radius > 400:
        return np.array([])

    h, w = objects_labeled.shape
    x, y = center_of_mass(obj_i)
    x, y = int(x), int(y)
    r1 = x-radius if x-radius >= 0 else 0
    r2 = x+radius if x+radius <= h else h
    c1 = y-radius if y-radius >= 0 else 0
    c2 = y+radius if y+radius < w else w
    indices = np.unique(objects_labeled[r1:r2, c1:c2])
    indices = indices[indices != 0]

    if indices.size == 0:
        indices = find_candidates(obj_i, objects_labeled, 2*radius)

    return indices


def AJI_fast(gt, pred_arr):
    gs, g_areas = np.unique(gt, return_counts=True)
    assert np.all(gs == np.arange(len(gs)))
    ss, s_areas = np.unique(pred_arr, return_counts=True)
    assert np.all(ss == np.arange(len(ss)))

    if len(ss) < 2:
        # All background.
        assert len(ss) == 1
        if len(gs) == 1:
            return 1.0
        else:
            return 0.0

    i_idx, i_cnt = np.unique(np.concatenate([gt.reshape(1, -1), pred_arr.reshape(1, -1)]),
                             return_counts=True, axis=1)
    i_arr = np.zeros(shape=(len(gs), len(ss)), dtype=np.int16)
    i_arr[i_idx[0], i_idx[1]] += i_cnt
    u_arr = g_areas.reshape(-1, 1) + s_areas.reshape(1, -1) - i_arr
    iou_arr = 1.0 * i_arr / u_arr

    i_arr = i_arr[1:, 1:]
    u_arr = u_arr[1:, 1:]
    iou_arr = iou_arr[1:, 1:]

    j = np.argmax(iou_arr, axis=1)

    c = np.sum(i_arr[np.arange(len(gs) - 1), j])
    u = np.sum(u_arr[np.arange(len(gs) - 1), j])
    used = np.zeros(shape=(len(ss) - 1), dtype=np.int16)
    used[j] = 1
    u += (np.sum(s_areas[1:] * (1 - used)))
    return 1.0 * c / u