import argparse
import os
import torch
import torch.nn.functional as F
import torchvision

from networks.network import Generator128x128


def _get_grid_size(num_images, max_len=20):
    '''Returns the number of images that will be put in a grid of size 8 x num_images.
       Grid size can be at most 8 x max_len.'''
    if num_images == 0 or num_images == 1:
        return 1
    elif num_images > max_len:
        return max_len *  max_len
    else:
        return num_images * num_images

PARSER = argparse.ArgumentParser()
PARSER.add_argument('--model_file_name',
                    default='final-128x128-generator.pth',
                    help='The file name of a trained model.')
PARSER.add_argument('--num_images',
                    default='8', type=int,
                    help='e.g. 15 -> grid of 15*15 images..')
args = PARSER.parse_args()

GRID_SIZE = _get_grid_size(args.num_images)
MODEL_PATH = f'trained_models/{args.model_file_name}'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_IMAGE_DIR = f'generated_with_preloaded_models/{args.model_file_name}'

if args.num_images > 1:
    # Save grids to separate directory.
    SAVE_IMAGE_DIR += '/grids'
else:
    # Add single images to a dedicated directory to use for evaluation.
    SAVE_IMAGE_DIR += '/1x1'


if not os.path.exists(SAVE_IMAGE_DIR):
    os.makedirs(SAVE_IMAGE_DIR)
IMAGE_NUM_SO_FAR = len(os.listdir(SAVE_IMAGE_DIR))

generator_model = Generator128x128().to(DEVICE)

# Deactivate residual elements in the generator.
generator_model.residual_rgb_conv = None
generator_model.residual_influence = None

generator_model.load_state_dict(torch.load(MODEL_PATH))
generator_model.eval()
print(f'Loaded model {MODEL_PATH}')

# Create a random batch of latent space vectors.
fixed_latent_space_vectors = torch.randn([GRID_SIZE, 512], device=DEVICE)

generated_images = generator_model(fixed_latent_space_vectors).detach()
generated_images = F.interpolate(generated_images, size=(128, 128), mode='nearest')
torchvision.utils.save_image(generated_images, f'{SAVE_IMAGE_DIR}/{(IMAGE_NUM_SO_FAR+1):03d}.jpg', padding=2, normalize=True)
print(f'Saved image {SAVE_IMAGE_DIR}/{(IMAGE_NUM_SO_FAR+1):03d}.jpg')
