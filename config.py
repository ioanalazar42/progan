'''Define different configurations with custom parameters to facilitate running experiments.
    Configurations can be specified as a command line argument.'''

_CONFIGURATIONS = {
    "default": {
        data_dir: '/home/datasets/celeba-aligned',
        save_image_dir: None,
        save_model_dir: None,
        tensorboard_dir: None,
        dry_run: False,
        model_save_frequency: 4,
        training_set_size: 99999999,
        gradient_penalty_factor: 10,
        learning_rate: 1e-4,
        mini_batch_size: 2,
        num_critic_training_steps: 2
        num_epochs: 20,
        transition_length_per_network: {
            4: 500,
            8: 2500,
            16: 5000,
            32: 10000,
            64: 20000,
            128: 40000
        }
        epoch_length_per_network: {
            4: 1000,
            8: 5000,
            16: 10000,
            32: 20000,
            64: 40000,
            128: 80000
        }
    },
}

def get_config(config_name):
    return _CONFIGURATIONS[config_name]
