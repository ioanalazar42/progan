import argparse
import numpy as np
import os
import time
import torch
import torch.nn.functional as F
import torchvision

from networks.network import Generator128x128


PARSER = argparse.ArgumentParser()
PARSER.add_argument('--model_file_name',
                    default='final-128x128-generator.pth',
                    help='The file name of a trained model.')
args = PARSER.parse_args()

MODEL_PATH = f'trained_models/{args.model_file_name}'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_IMAGE_DIR = f'generated_with_preloaded_models/{args.model_file_name}'

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
random_state = np.random.Generator(np.random.PCG64(np.random.SeedSequence(44442222)))
random_values = random_state.standard_normal([64, 512], dtype=np.float32)
fixed_latent_space_vectors = torch.tensor(random_values, device=DEVICE)

generated_images = generator_model(fixed_latent_space_vectors).detach()
generated_images = F.interpolate(generated_images, size=(128, 128), mode='nearest')
torchvision.utils.save_image(generated_images, f'{SAVE_IMAGE_DIR}/{(IMAGE_NUM_SO_FAR+1):03d}.jpg', padding=2, normalize=True)
print(f'Saved image {SAVE_IMAGE_DIR}/{(IMAGE_NUM_SO_FAR+1):03d}.jpg')
