import argparse
import numpy as np
import os
import time
import torch
import torch.nn.functional as F
import torchvision

from networks.network import Critic128x128


# Directory that contains single 128x128 images generated using pretrained models. 
IMAGE_DIR_PATH = '/home/ioanalazar459/progan/generated_with_preloaded_models/generator-128x128-12H.pth/1x1'

PARSER = argparse.ArgumentParser()
PARSER.add_argument('--model_file_name',
                    default='final-128x128-critic.pth',
                    help='The file name of a trained model.')
PARSER.add_argument('--image_path',
                    default='{IMAGE_DIR_PATH}/001.jpg',
                    help='The path to an image.')
args = PARSER.parse_args()

MODEL_PATH = f'trained_models/{args.model_file_name}'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set up pretrained Critic.
critic_model = Critic128x128().to(DEVICE)

# Deactivate residual elements in the critic.
critic_model.residual_rgb_conv = None
critic_model.residual_influence = None

critic_model.load_state_dict(torch.load(MODEL_PATH))
critic_model.eval()
print(f'Loaded model {MODEL_PATH}')

# Load the image and give it a score.
image = io.imread(args.image_path).transpose(2, 0, 1)
print('Loaded image {args.image_path}.')
print(f'Critic score: {critic_model(image)}.')
