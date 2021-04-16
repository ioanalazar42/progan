import argparse
import os
import torch
import torch.nn.functional as F
import torchvision

from networks.network5 import Generator128x128


PARSER = argparse.ArgumentParser()
PARSER.add_argument('--model_file_name',
                    default='deep-generator-128x128.pth',
                    help='The file name of a trained model')
PARSER.add_argument('--grid_size',
                    default='64', type=int,
                    help='Grid size -> [1, 64]')
PARSER.add_argument('--lsv_size',
                    default='512', type=int,
                    help='Size of latent space vectors')
PARSER.add_argument('--iterations',
                    default='1', type=int,
                    help='How many samples to generate')
args = PARSER.parse_args()

GRID_SIZE = min(64, max(1, args.grid_size))
MODEL_PATH = f'trained_models/{args.model_file_name}'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_IMAGE_DIR = f'generated_with_preloaded_models'

if args.grid_size > 1:
    # Save grids to separate directory.
    SAVE_IMAGE_DIR += '/grids'
else:
    # Add single images to a dedicated directory to use for evaluation.
    SAVE_IMAGE_DIR += '/1x1'


if not os.path.exists(SAVE_IMAGE_DIR):
    os.makedirs(SAVE_IMAGE_DIR)

generator_model = Generator128x128().to(DEVICE)

# Deactivate residual elements in the generator.
generator_model.residual_rgb_conv = None
generator_model.residual_influence = None

generator_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
generator_model.eval()
print(f'Loaded model "{MODEL_PATH}"')

for i in range(args.iterations):
    image_num_so_far = len(os.listdir(SAVE_IMAGE_DIR))

    # Create a random batch of latent space vectors.
    fixed_latent_space_vectors = torch.randn([GRID_SIZE, args.lsv_size], device=DEVICE)

    generated_images = generator_model(fixed_latent_space_vectors).detach()
    generated_images = F.interpolate(generated_images, size=(128, 128), mode='nearest')
    torchvision.utils.save_image(generated_images, f'{SAVE_IMAGE_DIR}/{(image_num_so_far+1):03d}.jpg', padding=2, normalize=True)
    print(f'Saved image "{SAVE_IMAGE_DIR}/{(image_num_so_far+1):03d}.jpg"')
