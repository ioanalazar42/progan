'''Contains utilities for reading and processing images.'''

import logging
import numpy as np
import os

from skimage import io, transform


def _mean_normalize(image):
    '''Takes an image with float values between [0, 255] and normalizes it to [-1, 1]'''
    return 2 * (image/255) - 1


def _load_image(path):
    image = io.imread(path)

    if image.ndim == 2:
        # Convert grayscale images to RGB
        print(f'Image "{path}" is grayscale!')
        image = np.dstack([image, image, image])

    image = _mean_normalize(image)

    # Change the image_size x image_size x 3 image to 3 x image_size x image_size as expected by PyTorch.
    return image.transpose(2, 0, 1)


def load_images(dir_path, training_set_size, image_size):
    logger = logging.getLogger()

    file_names = os.listdir(dir_path)[:training_set_size]
    images = np.empty([len(file_names), 3, image_size, image_size], dtype=np.float32)
    logger.info(f'Loading {len(file_names)} images of size {image_size}x{image_size} from {dir_path}...')

    for i, file_name in enumerate(file_names):
        image_path = os.path.join(dir_path, file_name)
        images[i] = _load_image(image_path)

        if i > 0 and i % 10000 == 0:
            logger.info(f'Loaded {i}/{len(images)} images so far')

    return images
