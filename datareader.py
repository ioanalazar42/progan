'''Contains utilities for reading and processing images.'''

import numpy as np
import os

from skimage import io, transform


def _center_crop_image(image):
    height = image.shape[0]
    width = image.shape[1]
    crop_size = height if height < width else width

    y = int((height - crop_size) / 2)
    x = int((width - crop_size) / 2)

    return image[y : crop_size, x : crop_size]


def _resize_image(image, width, height):
    return transform.resize(image, [height, width, 3], anti_aliasing=True, mode='constant')


def _mean_normalize(image):
    '''Takes an image with float values between [0, 1] and normalizes it to [-1, 1]'''
    return 2 * image - 1


def _load_image(path, image_size):
    image = io.imread(path)

    if image.ndim == 2:
        # Convert grayscale images to RGB
        # (yes you can actually save a JPEG without RGB info (so like 500x500 instead of 500x500x3), *mind blown*)
        print(f'Image "{path}" is grayscale!')
        image = np.dstack([image, image, image])

    image = _mean_normalize(_resize_image(_center_crop_image(image), image_size, image_size))

    # Change the image_size x image_size x 3 image to 3 x image_size x image_size as expected by PyTorch.
    return image.transpose(2, 0, 1)


def load_images(dir_path, training_set_size, image_size):
    file_names = os.listdir(dir_path)[:training_set_size]
    images = np.empty([len(file_names), 3, image_size, image_size], dtype=np.float32)
    print(f'\nLoading {len(file_names)} images from {dir_path}...\n')

    for i, file_name in enumerate(file_names):
        image_path = os.path.join(dir_path, file_name)
        images[i] = _load_image(image_path, image_size)

        if i > 0 and i % 10000 == 0:
            print(f'Loaded {i}/{len(images)} images so far')

    return images
