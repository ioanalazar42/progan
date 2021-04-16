import argparse
import numpy as np
import os
import random
import torch

from networks.network5 import Critic128x128
from skimage import io


def _score_image(image, critic_model, device):
    score = critic_model(torch.tensor(image, device=device)).item()
    print(f'Critic score: {score:.3f}\n')

def _load_image(path):
    image = io.imread(path)
    print(f'Loaded image "{path}".')
    image = 2 * (image/255) - 1 # Mean normalize image.
    image = np.expand_dims(image.transpose(2, 0, 1), axis=0) # [img_size, img_size, 3] -> [3, img_size, img_size] -> [1, 3, img_size, img_size]
    return image.astype(np.float32)


# Directory that contains single 128x128 images generated using pretrained models. 
IMAGE_DIR_PATH = 'generated_images/1x1'

PARSER = argparse.ArgumentParser()
PARSER.add_argument('--model_file_name',
                    default='deep-critic-128x128.pth',
                    help='The file name of a trained model.')
PARSER.add_argument('--image_path',
                    default=f'random',
                    help='The path to an image.')
PARSER.add_argument('--iterations',
                    default='1', type=int,
                    help='How many random images to score')
args = PARSER.parse_args()

MODEL_PATH = f'trained_models/{args.model_file_name}'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set up pretrained Critic.
critic_model = Critic128x128().to(DEVICE)

# Deactivate residual elements in the critic.
critic_model.residual_rgb_conv = None
critic_model.residual_influence = None

critic_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
critic_model.eval()
print(f'Loaded model "{MODEL_PATH}"')

if args.image_path == 'random':
    print(f'Picking {args.iterations} random images.\n')
    file_names = np.asarray(os.listdir(IMAGE_DIR_PATH))
    random_indexes = np.random.choice(len(file_names), args.iterations)
    file_names = file_names[random_indexes]

    for i, file_name in enumerate(file_names):
        image_path = os.path.join(IMAGE_DIR_PATH, file_name)
        _score_image(_load_image(image_path), critic_model, DEVICE)
else:
    _score_image(_load_image(args.image_path), critic_model, DEVICE)
