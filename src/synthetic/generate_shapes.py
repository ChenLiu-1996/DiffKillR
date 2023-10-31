import numpy as np
from typing import Tuple


def _generate_square(image_shape: Tuple[int] = (64, 64), edge_length: int = 32, random_seed: int = None):
    image = np.zeros(image_shape, dtype=np.float32)

    if random_seed is not None:
        np.random.seed(random_seed)
    top_left_corner = [int(np.random.uniform(0, image_shape[i]//2)) for i in range(2)]

    image[top_left_corner[0]:top_left_corner[0] + edge_length,
          top_left_corner[1]:top_left_corner[1] + edge_length] = 1.0

    return image

def generate_shape(image_shape: Tuple[int] = (64, 64), shape: str = 'square', random_seed: int = None):
    if shape == 'square':
        image = _generate_square(image_shape=image_shape, random_seed=random_seed)

    return image

if __name__ == '__main__':
    image_square = _generate_square()
