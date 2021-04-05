import argparse
import logging
import numpy as np
import os
import pprint
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from config import get_configuration
from datareader import load_images
from network import Critic4x4, Generator4x4
from timeit import default_timer as timer
from torch.utils import tensorboard
from utils import configure_logger, sample_gradient_l2_norm

PARSER = argparse.ArgumentParser()
PARSER.add_argument('--configuration',
                    default='default',
                    help='The name of a configuration that defines parameters like: size of training set, number of epochs etc.')
args = PARSER.parse_args()

# Define constants.
EXPERIMENT_ID = int(time.time()) # Used to create new directories to save results of individual experiments.

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONFIG = get_configuration(args.configuration) # Get the current configuration.
SAVE_IMAGE_DIR = CONFIG.get('save_image_dir', default=f'images/{EXPERIMENT_ID}-{args.configuration}')
TENSORBOARD_DIR = CONFIG.get('tensorboard_dir', default=f'tensorboard/{EXPERIMENT_ID}-{args.configuration}')
SAVE_MODEL_DIR = CONFIG.get('save_model_dir', default=f'models/{EXPERIMENT_ID}-{args.configuration}')
SAVE_LOGS_DIR = CONFIG.get('save_logs_dir', default=f'logs')

# Set up logging of information. Will print both to console and a file that has this format: 'logs/<EXPERIMENT_ID>.log'
logger = logging.getLogger()
configure_logger(SAVE_LOGS_DIR, EXPERIMENT_ID, args.configuration)

if CONFIG.is_enabled('dry_run'):
    logger.info('Dry run! Just for testing, data is not saved')

if CONFIG.missing_fields():
    raise Exception(f'Configuration {args.configuration} misses the following fields: {CONFIG.missing_fields()}\n')

# Create directories for images, tensorboard results, saved models and experiment logs.
if CONFIG.is_disabled('dry_run'):
    os.makedirs(SAVE_IMAGE_DIR)
    os.makedirs(TENSORBOARD_DIR)
    os.makedirs(SAVE_MODEL_DIR)
    WRITER = tensorboard.SummaryWriter(TENSORBOARD_DIR) # Set up TensorBoard.

# Print what configuration is being used and let user know whether or not data is saved for this experiment.
logger.info(f'Using configuration "{args.configuration}".')
logger.info(pprint.pformat(CONFIG.to_dictionary(), indent=4))

# Create a random batch of latent space vectors that will be used to visualize the progression of the generator.
# Use the same values (seeded at 44442222) between multiple runs, so that the progression can still be seen when loading saved models.
random_state = np.random.Generator(np.random.PCG64(np.random.SeedSequence(44442222)))
random_values = random_state.standard_normal([64, 512], dtype=np.float32)
fixed_latent_space_vectors = torch.tensor(random_values, device=DEVICE)

global_epoch_count = 0

for network_size in [4, 8, 16, 32, 64, 128]:
    # Deallocate previous images.
    images = None

    # Load and preprocess images:
    images = load_images(CONFIG.get('data_dir_per_network_size')[network_size], CONFIG.get('training_set_size'), image_size=network_size)

    print()
    if network_size == 4:
        critic_model = Critic4x4().to(DEVICE)
        generator_model = Generator4x4().to(DEVICE)
    else:
        logger.info(f'BEGIN TRANSITIONING TO {network_size}x{network_size}.')
        critic_model = critic_model.evolve(DEVICE)
        generator_model = generator_model.evolve(DEVICE)
    
    # Set up Adam optimizers for both models.
    critic_optimizer = optim.Adam(critic_model.parameters(), lr=CONFIG.get('learning_rate'), betas=(0, 0.99))
    generator_optimizer = optim.Adam(generator_model.parameters(), lr=CONFIG.get('learning_rate'), betas=(0, 0.99))
    
    # When the transition to a larger network is finished, log this.
    logged_transition_finished = False
    
    for epoch in range(CONFIG.get('num_epochs_per_network')[network_size]):
        start_time = timer()

        # Variables for recording statistics.
        average_critic_real_performance = 0.0  # C(x) - The critic wants this to be as big as possible for real images.
        average_critic_generated_performance = 0.0  # C(G(x)) - The critic wants this to be as small as possible for generated images.
        average_critic_loss = 0.0
        average_generator_loss = 0.0

        # Train: For every epoch, perform the number of mini-batch updates that corresonds to the current network size.
        for _ in range(CONFIG.get('epoch_length_per_network')[network_size]):

            if network_size > 4:
                if critic_model.residual_influence > 0:
                    critic_model.residual_influence -= 1 / CONFIG.get('transition_length_per_network')[network_size]
                    generator_model.residual_influence -= 1 / CONFIG.get('transition_length_per_network')[network_size]
            
                elif not logged_transition_finished: # Residual influence zero, so finished transitioning.
                    logger.info(f'FINISHED TRANSITIONING TO {network_size}x{network_size}.')
                    logged_transition_finished = True

            # Train the critic:
            for _ in range(CONFIG.get('num_critic_training_steps')):
                critic_model.zero_grad()

                # Evaluate a mini-batch of real images.
                random_indexes = np.random.choice(len(images), CONFIG.get('mini_batch_size'))
                real_images = torch.tensor(images[random_indexes], device=DEVICE)

                real_scores = critic_model(real_images)

                # Evaluate a mini-batch of generated images.
                random_latent_space_vectors = torch.randn(CONFIG.get('mini_batch_size'), 512, device=DEVICE)
                generated_images = generator_model(random_latent_space_vectors)

                generated_scores = critic_model(generated_images.detach()) # TODO: Why exactly is `.detach` required?

                gradient_l2_norm = sample_gradient_l2_norm(critic_model, real_images, generated_images, DEVICE)
                
                # Update the weights.
                loss = torch.mean(generated_scores) - torch.mean(real_scores) + CONFIG.get('gradient_penalty_factor') * gradient_l2_norm  # The critic's goal is for `generated_scores` to be small and `real_scores` to be big.
                loss.backward()
                critic_optimizer.step()

                # Record some statistics.
                average_critic_loss += loss.item() / CONFIG.get('num_critic_training_steps') / CONFIG.get('epoch_length_per_network')[network_size]
                average_critic_real_performance += real_scores.mean().item() / CONFIG.get('num_critic_training_steps') / CONFIG.get('epoch_length_per_network')[network_size]
                average_critic_generated_performance += generated_scores.mean().item() / CONFIG.get('num_critic_training_steps') / CONFIG.get('epoch_length_per_network')[network_size]
                discernability_score = average_critic_real_performance - average_critic_generated_performance # Measure how different generated images are from real images. This should trend towards 0 as fake images become indistinguishable from real ones to the critic.

            # Train the generator:
            generator_model.zero_grad()
            generated_scores = critic_model(generated_images)

            # Update the weights.
            loss = -torch.mean(generated_scores)  # The generator's goal is for `generated_scores` to be big.
            loss.backward()
            generator_optimizer.step()

            # Record some statistics.
            average_generator_loss += loss.item() / CONFIG.get('epoch_length_per_network')[network_size]
     
        # Record statistics.
        global_epoch_count += 1
        time_elapsed = timer() - start_time

        # Log some statistics in the terminal.
        logger.info(f'{network_size}x{network_size} | {epoch:3} | '
            f'Loss(C): {average_critic_loss:.6f} | '
            f'Loss(G): {average_generator_loss:.6f} | '
            f'Avg C(x): {average_critic_real_performance:.6f} | '
            f'Avg C(G(x)): {average_critic_generated_performance:.6f} | '
            f'C(x) - C(G(x)): {discernability_score:.6f} | '
            f'Time: {time_elapsed:.3f}s')

        # Save the model parameters at a specified interval.
        if (CONFIG.is_disabled('dry_run') 
            and global_epoch_count > 0 
            and (global_epoch_count % CONFIG.get('model_save_frequency') == 0 or epoch == CONFIG.get('num_epochs_per_network')[network_size] - 1)):
            
            save_critic_model_path = f'{SAVE_MODEL_DIR}/critic-{network_size}x{network_size}-{epoch}.pth'
            logger.info(f'\nSaving critic model as "{save_critic_model_path}"...')
            torch.save(critic_model.state_dict(), save_critic_model_path)
            
            save_generator_model_path = f'{SAVE_MODEL_DIR}/generator-{network_size}x{network_size}-{epoch}.pth'
            logger.info(f'Saving generator model as "{save_generator_model_path}"...\n')
            torch.save(generator_model.state_dict(), save_generator_model_path)

        # Save generated images (and real images if they are required for comparison).
        if CONFIG.is_disabled('dry_run'):
            # Check if the experiment requires the saving of real images for comparison purposes.
            if CONFIG.is_enabled('save_real_images'):
                random_indexes = np.random.choice(len(images), 64)
                with torch.no_grad():
                    real_images = torch.tensor(images[random_indexes], device=DEVICE)
                real_images = F.interpolate(real_images, size=(128, 128), mode='nearest')
                torchvision.utils.save_image(real_images, f'{SAVE_IMAGE_DIR}/{global_epoch_count:03d}-{network_size}x{network_size}-{epoch}-real.jpg', padding=2, normalize=True)
            
            with torch.no_grad():
                generated_images = generator_model(fixed_latent_space_vectors).detach()

            generated_images = F.interpolate(generated_images, size=(128, 128), mode='nearest')
            grid_images = torchvision.utils.make_grid(generated_images, padding=2, normalize=True)
            torchvision.utils.save_image(generated_images, f'{SAVE_IMAGE_DIR}/{global_epoch_count:03d}-{network_size}x{network_size}-{epoch}.jpg', padding=2, normalize=True)

            WRITER.add_image('training/generated-images', grid_images, global_epoch_count)
            WRITER.add_scalar('training/generator/loss', average_generator_loss, global_epoch_count)
            WRITER.add_scalar('training/critic/loss', average_critic_loss, global_epoch_count)
            WRITER.add_scalar('training/critic/real-performance', average_critic_real_performance, global_epoch_count)
            WRITER.add_scalar('training/critic/generated-performance', average_critic_generated_performance, global_epoch_count)
            WRITER.add_scalar('training/discernability-score', discernability_score, global_epoch_count)
            WRITER.add_scalar('training/epoch-duration', time_elapsed, global_epoch_count)

logger.info('Finished training!')
