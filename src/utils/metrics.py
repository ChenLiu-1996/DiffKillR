import numpy as np
from skimage.metrics import structural_similarity


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