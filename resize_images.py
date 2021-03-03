'''Code that can be run once to resize images for the ProGAN program.

Resize images in the CelebA dataset to prepare them for different network architectures, i.e. 4x4, 8x8, ...
Save images in separate directories that correspond to their new sizes.
'''

import numpy as np
import os
import torchvision

from skimage import io, transform

TRAINING_IMAGES_DIR_PATH = '/home/datasets/'
file_names = os.listdir(TRAINING_IMAGES_DIR_PATH)

images = []
print(f'\nLoading {len(file_names)} images from {dir_path}...\n')

# Load all images in their original size.
for i, file_name in enumerate(file_names):
    image_path = os.path.join(dir_path, file_name)
    images.append(_load_image(image_path))

    if i > 0 and i % 10000 == 0:
        print(f'Loaded {i}/{len(images)} images so far')

# Resize all images and save them in separate directories.
for image_size in [4, 8, 16, 32, 64, 128]:
    save_image_dir = f'{TRAINING_IMAGES_DIR_PATH}/celeba-{image_size}x{image_size}'
    os.makedirs(save_image_dir)

    file_id = 1
        
    for image in images:
        resized_image = _resize_image(image)
        torchvision.utils.save_image(resized_image, f'{save_image_dir}/{file_id:06d}.jpg', padding=2, normalize=True)
        file_id += 1


def _center_crop_image(image):
    height = image.shape[0]
    width = image.shape[1]
    crop_size = height if height < width else width

    y = int((height - crop_size) / 2)
    x = int((width - crop_size) / 2)

    return image[y : crop_size, x : crop_size]


def _resize_image(image, width, height):
    return transform.resize(image, [height, width, 3], anti_aliasing=True, mode='constant')


def _load_image(path):
    image = io.imread(path)

    if image.ndim == 2:
        # Convert grayscale images to RGB
        print(f'Image "{path}" is grayscale!')
        image = np.dstack([image, image, image])

    image = _center_crop_image(image)

    return image
