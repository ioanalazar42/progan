import argparse
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from datareader import load_images
from network import Critic4x4, Generator4x4
from timeit import default_timer as timer
from torch.utils.tensorboard import SummaryWriter
from utils import sample_gradient_l2_norm

EXPERIMENT_ID = int(time.time()) # Used to create new directories to save results of individual experiments
# Directories to save results of experiments.
DEFAULT_IMG_DIR = 'images/{}'.format(EXPERIMENT_ID)
DEFAULT_TENSORBOARD_DIR = 'tensorboard/{}'.format(EXPERIMENT_ID)
DEFAULT_MODEL_DIR = 'models/{}'.format(EXPERIMENT_ID)


PARSER = argparse.ArgumentParser()

PARSER.add_argument('--data_dir', default='/home/datasets/celeba-aligned')
PARSER.add_argument('--load_critic_model_path')
PARSER.add_argument('--load_generator_model_path')
PARSER.add_argument('--save_image_dir', default=DEFAULT_IMG_DIR)
PARSER.add_argument('--save_model_dir', default=DEFAULT_MODEL_DIR)
PARSER.add_argument('--tensorboard_dir', default=DEFAULT_TENSORBOARD_DIR)
PARSER.add_argument('--dry_run', default=False, type=bool)
PARSER.add_argument('--model_save_frequency', default=4, type=int)

PARSER.add_argument('--training_set_size', default=99999999, type=int)
PARSER.add_argument('--epoch_length', default=2500, type=int)
PARSER.add_argument('--gradient_penalty_factor', default=10, type=float)
PARSER.add_argument('--learning_rate', default=0.0001, type=float)
PARSER.add_argument('--mini_batch_size', default=2, type=int)
PARSER.add_argument('--num_critic_training_steps', default=2, type=int)
PARSER.add_argument('--num_epochs', default=20, type=int)
PARSER.add_argument('--transition_length', default=25000, type=int)

args = PARSER.parse_args()

# Create directories for images, tensorboard results and saved models.
if not args.dry_run:
    os.makedirs(args.save_image_dir)
    os.makedirs(args.tensorboard_dir)
    os.makedirs(args.save_model_dir)
else:
    print('Dry run! Just for testing, data is not saved')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
total_training_steps = 0

# Create a random batch of latent space vectors that will be used to visualize the progression of the generator.
fixed_latent_space_vectors = torch.randn(64, 512, device=device)  # Note: randn is sampling from a normal distribution

# Set up TensorBoard.
writer = SummaryWriter(args.tensorboard_dir)

for network_size in [4, 8, 16, 32, 64, 128]:
    # Deallocate previous images.
    images = None

    # Load and preprocess images:
    images = load_images(args.data_dir, args.training_set_size, network_size)

    if network_size == 4:
        critic_model = Critic4x4().to(device)
        generator_model = Generator4x4().to(device)
    else:
        critic_model = critic_model.evolve()
        generator_model = generator_model.evolve()
    
    # Set up Adam optimizers for both models.
    critic_optimizer = optim.Adam(critic_model.parameters(), lr=args.learning_rate, betas=(0, 0.9))
    generator_optimizer = optim.Adam(generator_model.parameters(), lr=args.learning_rate, betas=(0, 0.9))

    for epoch in range(args.num_epochs):
        start_time = timer()
        
        # Variables for recording statistics.
        average_critic_real_performance = 0.0  # C(x) - The critic wants this to be as big as possible for real images.
        average_critic_generated_performance = 0.0  # C(G(x)) - The critic wants this to be as small as possible for generated images.
        average_critic_loss = 0.0
        average_generator_loss = 0.0

        # Train: perform `args.epoch_length` mini-batch updates per "epoch".
        for i in range(args.epoch_length):

            if network_size > 4 and critic_model.residual_influence > 0:
                critic_model.residual_influence -= 1 / args.transition_length
                generator_model.residual_influence -= 1 / args.transition_length

            # Train the critic:
            for i in range(args.num_critic_training_steps):
                critic_model.zero_grad()

                # Evaluate a mini-batch of real images.
                random_indexes = np.random.choice(len(images), args.mini_batch_size)
                real_images = torch.tensor(images[random_indexes], device=device)

                real_scores = critic_model(real_images)

                # Evaluate a mini-batch of generated images.
                random_latent_space_vectors = torch.randn(args.mini_batch_size, 512, device=device)
                generated_images = generator_model(random_latent_space_vectors)

                generated_scores = critic_model(generated_images.detach()) # TODO: Why exactly is `.detach` required?

                gradient_l2_norm = sample_gradient_l2_norm(critic_model, real_images, generated_images, device)
                
                # Update the weights.
                loss = torch.mean(generated_scores) - torch.mean(real_scores) + args.gradient_penalty_factor * gradient_l2_norm  # The critic's goal is for `generated_scores` to be small and `real_scores` to be big. I don't know why we had to overcomplicate things and call this "Wasserstein".
                loss.backward()
                critic_optimizer.step()

                # Record some statistics.
                average_critic_loss += loss.item() / args.num_critic_training_steps / args.epoch_length
                average_critic_real_performance += real_scores.mean().item() / args.num_critic_training_steps / args.epoch_length
                average_critic_generated_performance += generated_scores.mean().item() / args.num_critic_training_steps / args.epoch_length

            # Train the generator:
            generator_model.zero_grad()
            generated_scores = critic_model(generated_images)

            # Update the weights.
            loss = -torch.mean(generated_scores)  # The generator's goal is for `generated_scores` to be big.
            loss.backward()
            generator_optimizer.step()

            # Record some statistics.
            average_generator_loss += loss.item() / args.epoch_length
     
        # Record statistics.
        total_training_steps += 1
        time_elapsed = timer() - start_time

        print(('Network size: {} - Epoch: {} - Critic Loss: {:.6f} - Generator Loss: {:.6f} - Average C(x): {:.6f} - Average C(G(x)): {:.6f} - Time: {:.3f}s')
            .format(network_size, epoch, average_critic_loss , average_generator_loss, average_critic_real_performance, average_critic_generated_performance, time_elapsed))
        
        # Save the model parameters at a specified interval.
        if (not args.dry_run 
            and epoch > 0 
            and (epoch % args.model_save_frequency == 0 or epoch == args.num_epochs - 1)):
            save_critic_model_path = '{}/critic_{}-{}x{}-{}.pth'.format(args.save_model_dir, network_size, network_size, EXPERIMENT_ID, epoch)
            print('\nSaving critic model as "{}"...'.format(save_critic_model_path))
            torch.save(critic_model.state_dict(), save_critic_model_path)
            
            save_generator_model_path = '{}/generator_{}-{}x{}-{}.pth'.format(args.save_model_dir, network_size, network_size, EXPERIMENT_ID, epoch)
            print('Saving generator model as "{}"...\n'.format(save_generator_model_path,))
            torch.save(generator_model.state_dict(), save_generator_model_path)

        # Save images.
        if not args.dry_run:
            with torch.no_grad():
                generated_images = generator_model(fixed_latent_space_vectors).detach()
            generated_images = F.interpolate(generated_images, scale_factor= 128 / network_size, mode='nearest')
            grid_images = torchvision.utils.make_grid(generated_images, padding=2, normalize=True)
            torchvision.utils.save_image(generated_images, '{}/{:03d}-{}x{}-{}.jpg'.format(args.save_image_dir, total_training_steps, network_size, network_size, epoch), padding=2, normalize=True)

            writer.add_image('training/generated-images', grid_images, epoch)
        writer.add_scalar('training/generator/loss', average_generator_loss, epoch)
        writer.add_scalar('training/critic/loss', average_critic_loss, epoch)
        writer.add_scalar('training/critic/real-performance', average_critic_real_performance, epoch)
        writer.add_scalar('training/critic/generated-performance', average_critic_generated_performance, epoch)
        writer.add_scalar('training/epoch-duration', time_elapsed, epoch)

print('Finished training!')

