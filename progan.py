import argparse
import numpy as np
import os
import pprint
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from config import get_configuration
from datareader import load_images
from network import Critic4x4, Generator4x4
from timeit import default_timer as timer
from torch.utils.tensorboard import SummaryWriter
from utils import sample_gradient_l2_norm


# Used to create new directories to save results of individual experiments.
EXPERIMENT_ID = int(time.time()) 

# Directories to save results of experiments.
DEFAULT_IMG_DIR = f'images/{EXPERIMENT_ID}'
DEFAULT_TENSORBOARD_DIR = f'tensorboard/{EXPERIMENT_ID}'
DEFAULT_MODEL_DIR = f'models/{EXPERIMENT_ID}'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRETTY_PRINTER = pprint.PrettyPrinter(indent=4)

# The running configuration can be passed as a command line argument. The configuration defines parameters like: 
# size of training set, number of epochs etc. 
PARSER = argparse.ArgumentParser()
PARSER.add_argument('--configuration', default='default')
args = PARSER.parse_args()

config = get_configuration(args.configuration) # Get the current configuration.

missing_fields = config.missing_fields()
if missing_fields:
    raise Exception(f'Configuration {args.configuration} misses the following fields: {missing_fields}\n')

print(f'Using configuration "{args.configuration}".\n')
PRETTY_PRINTER.pprint(config.to_dictionary())

# Set the directories to save models, images and tensorboard data according to the EXPERIMENT_ID. 
save_image_dir = config.get('save_image_dir', default=DEFAULT_IMG_DIR)
tensorboard_dir = config.get('tensorboard_dir', default=DEFAULT_TENSORBOARD_DIR)
save_model_dir = config.get('save_model_dir', default=DEFAULT_MODEL_DIR)

# Create directories for images, tensorboard results and saved models.
if config.is_disabled('dry_run'):
    os.makedirs(save_image_dir)
    os.makedirs(tensorboard_dir)
    os.makedirs(save_model_dir)
else:
    print('\nDry run! Just for testing, data is not saved')

# Create a random batch of latent space vectors that will be used to visualize the progression of the generator.
# Use the same values (seeded at 44442222) between multiple runs, so that the progression can still be seen when loading saved models.
random_state = np.random.Generator(np.random.PCG64(np.random.SeedSequence(44442222)))
random_values = random_state.standard_normal([64, 512], dtype=np.float32)
fixed_latent_space_vectors = torch.tensor(random_values, device=DEVICE)

# Set up TensorBoard.
writer = SummaryWriter(tensorboard_dir)

total_training_steps = 0
for network_size in [4, 8, 16, 32, 64, 128]:
    # Deallocate previous images.
    images = None

    # Load and preprocess images:
    images = load_images(config.get('data_dir_per_network_size')[network_size], config.get('training_set_size'), image_size=network_size)

    if network_size == 4:
        critic_model = Critic4x4().to(DEVICE)
        generator_model = Generator4x4().to(DEVICE)
    else:
        critic_model = critic_model.evolve(DEVICE)
        generator_model = generator_model.evolve(DEVICE)
    
    # Set up Adam optimizers for both models.
    critic_optimizer = optim.Adam(critic_model.parameters(), lr=config.get('learning_rate'), betas=(0, 0.99))
    generator_optimizer = optim.Adam(generator_model.parameters(), lr=config.get('learning_rate'), betas=(0, 0.99))

    for epoch in range(config.get('num_epochs_per_network')[network_size]):
        start_time = timer()
        
        # Variables for recording statistics.
        average_critic_real_performance = 0.0  # C(x) - The critic wants this to be as big as possible for real images.
        average_critic_generated_performance = 0.0  # C(G(x)) - The critic wants this to be as small as possible for generated images.
        average_critic_loss = 0.0
        average_generator_loss = 0.0

        # Train: For every epoch, perform the number of mini-batch updates that corresonds to the current network size.
        for i in range(config.get('epoch_length_per_network')[network_size]):

            if network_size > 4 and critic_model.residual_influence > 0:
                critic_model.residual_influence -= 1 / config.get('transition_length_per_network')[network_size]
                generator_model.residual_influence -= 1 / config.get('transition_length_per_network')[network_size]

            # Train the critic:
            for i in range(config.get('num_critic_training_steps')):
                critic_model.zero_grad()

                # Evaluate a mini-batch of real images.
                random_indexes = np.random.choice(len(images), config.get('mini_batch_size'))
                real_images = torch.tensor(images[random_indexes], device=DEVICE)

                real_scores = critic_model(real_images)

                # Evaluate a mini-batch of generated images.
                random_latent_space_vectors = torch.randn(config.get('mini_batch_size'), 512, device=DEVICE)
                generated_images = generator_model(random_latent_space_vectors)

                generated_scores = critic_model(generated_images.detach()) # TODO: Why exactly is `.detach` required?

                gradient_l2_norm = sample_gradient_l2_norm(critic_model, real_images, generated_images, DEVICE)
                
                # Update the weights.
                loss = torch.mean(generated_scores) - torch.mean(real_scores) + config.get('gradient_penalty_factor') * gradient_l2_norm  # The critic's goal is for `generated_scores` to be small and `real_scores` to be big. I don't know why we had to overcomplicate things and call this "Wasserstein".
                loss.backward()
                critic_optimizer.step()

                # Record some statistics.
                average_critic_loss += loss.item() / config.get('num_critic_training_steps') / config.get('epoch_length_per_network')[network_size]
                average_critic_real_performance += real_scores.mean().item() / config.get('num_critic_training_steps') / config.get('epoch_length_per_network')[network_size]
                average_critic_generated_performance += generated_scores.mean().item() / config.get('num_critic_training_steps') / config.get('epoch_length_per_network')[network_size]

            # Train the generator:
            generator_model.zero_grad()
            generated_scores = critic_model(generated_images)

            # Update the weights.
            loss = -torch.mean(generated_scores)  # The generator's goal is for `generated_scores` to be big.
            loss.backward()
            generator_optimizer.step()

            # Record some statistics.
            average_generator_loss += loss.item() / config.get('epoch_length_per_network')[network_size]
     
        # Record statistics.
        total_training_steps += 1
        time_elapsed = timer() - start_time

        # Print some statistics in the terminal.
        print(f'Network size: {network_size} - Epoch: {epoch} - '
            f'Critic Loss: {average_critic_loss:.6f} - '
            f'Generator Loss: {average_generator_loss:.6f} - '
            f'Average C(x): {average_critic_real_performance:.6f} - '
            f'Average C(G(x)): {average_critic_generated_performance:.6f} - '
            f'Time: {time_elapsed:.3f}s')

        # Save the model parameters at a specified interval.
        if (config.is_disabled('dry_run') 
            and epoch > 0 
            and (epoch % config.get('model_save_frequency') == 0 or epoch == config.get('num_epochs_per_network')[network_size] - 1)):
            
            save_critic_model_path = f'{save_model_dir}/critic-{network_size}x{network_size}-{epoch}.pth'
            print(f'\nSaving critic model as "{save_critic_model_path}"...')
            torch.save(critic_model.state_dict(), save_critic_model_path)
            
            save_generator_model_path = f'{save_model_dir}/generator-{network_size}x{network_size}-{epoch}.pth'
            print(f'Saving generator model as "{save_generator_model_path}"...\n')
            torch.save(generator_model.state_dict(), save_generator_model_path)

        # Save images.
        if config.is_disabled('dry_run'):
            with torch.no_grad():
                generated_images = generator_model(fixed_latent_space_vectors).detach()
            generated_images = F.interpolate(generated_images, size=(128, 128), mode='nearest')
            grid_images = torchvision.utils.make_grid(generated_images, padding=2, normalize=True)
            torchvision.utils.save_image(generated_images, f'{save_image_dir}/{total_training_steps:03d}-{network_size}x{network_size}-{epoch}.jpg', padding=2, normalize=True)

            writer.add_image('training/generated-images', grid_images, epoch)
        
        writer.add_scalar('training/generator/loss', average_generator_loss, epoch)
        writer.add_scalar('training/critic/loss', average_critic_loss, epoch)
        writer.add_scalar('training/critic/real-performance', average_critic_real_performance, epoch)
        writer.add_scalar('training/critic/generated-performance', average_critic_generated_performance, epoch)
        writer.add_scalar('training/epoch-duration', time_elapsed, epoch)

print('Finished training!')

