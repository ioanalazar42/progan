'''Code that can be run once to resize images for the ProGAN program.

Resize images in the CelebA dataset to prepare them for different network architectures, i.e. 4x4, 8x8, ...
Save images in separate directories that correspond to their new sizes.
'''

import argparse
import numpy as np
import os

from skimage import img_as_ubyte, io, transform

# Define constants for different modes of image resizing.
BILINEAR = 1
NEAREST_NEIGHBOR = 0
PARSER = argparse.ArgumentParser()
TRAINING_IMAGES_DIR_PATH = '/home/datasets/celeba-aligned' # Path to all training images.
                                                           # Will contain directories for images of different sizes to use for the ProGAN.
ORIGINAL_IMAGES_DIR_PATH = f'{TRAINING_IMAGES_DIR_PATH}/original' # The path to the original CelebA images.


def _center_crop_image(image):
    height = image.shape[0]
    width = image.shape[1]
    crop_size = height if height < width else width

    y = int((height - crop_size) / 2)
    x = int((width - crop_size) / 2)

    return image[y : crop_size, x : crop_size]


def _resize_image(image, width, height, mode):
    return transform.resize(image, [height, width, 3], order=mode, anti_aliasing=True, mode='constant')


def _load_image(path):
    image = io.imread(path)

    if image.ndim == 2:
        # Convert grayscale images to RGB
        print(f'Image "{path}" is grayscale!')
        image = np.dstack([image, image, image])

    image = _center_crop_image(image)

    return image


if __name__ == '__main__':
    PARSER.add_argument('--training_set_size', default=99999999, type=int)
    args = PARSER.parse_args()

    file_names = os.listdir(ORIGINAL_IMAGES_DIR_PATH)[:args.training_set_size]

    images = []
    print(f'\nLoading {len(file_names)} images from {ORIGINAL_IMAGES_DIR_PATH}...\n')

    # Load all images in their original size.
    for i, file_name in enumerate(file_names):
        image_path = os.path.join(ORIGINAL_IMAGES_DIR_PATH, file_name)
        images.append(_load_image(image_path))

        if i > 0 and i % 10000 == 0:
            print(f'Loaded {i}/{len(file_names)} images so far')

    # Resize all images and save them in separate directories.
    for image_size in [128, 64, 32, 16, 8, 4]:
        save_image_dir = f'{TRAINING_IMAGES_DIR_PATH}/new-{image_size}x{image_size}'
        os.makedirs(save_image_dir)

        for file_id, image in enumerate(images):
            resized_image = (_resize_image(image, image_size, image_size, mode=BILINEAR) if image_size==128
                             else _resize_image(image, image_size, image_size, mode=NEAREST_NEIGHBOR))  
            io.imsave(f'{save_image_dir}/{file_id:06d}.jpg', img_as_ubyte(resized_image))

            # Overwrite original images with resized in order to progressively shrink images (128x128 -> 64x64 -> 32x32 etc).
            images[file_id] = resized_image 
        
        print(f'\nLoaded {len(images)} images of size {image_size}x{image_size} in directory {save_image_dir}.\n')
