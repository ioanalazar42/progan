'''Defines configurations for experiments.

Defines different configurations with custom parameters to run experiments.
Configurations can be specified as a command line argument.
'''

_REQUIRED_FIELDS = [
    'network_type',
    'data_dir_per_network_size',
    'model_save_frequency',
    'image_save_frequency',
    'training_set_size',
    'gradient_penalty_factor',
    'learning_rate',
    'mini_batch_size',
    'num_critic_training_steps',
    'num_epochs_per_network',
    'transition_length_per_network',
    'epoch_length_per_network']

_CONFIGURATIONS = {
    'default': {
        'network_type': 'network',
        'data_dir_per_network_size': {
            4: '/home/datasets/celeba-aligned/4x4',
            8: '/home/datasets/celeba-aligned/8x8',
            16: '/home/datasets/celeba-aligned/16x16',
            32: '/home/datasets/celeba-aligned/32x32',
            64: '/home/datasets/celeba-aligned/64x64',
            128: '/home/datasets/celeba-aligned/128x128'
        },
        'dry_run': False,
        'model_save_frequency': 4,
        'image_save_frequency': 5000,
        'training_set_size': 99999999,
        'gradient_penalty_factor': 10,
        'learning_rate': 1e-4,
        'mini_batch_size': 256,
        'num_critic_training_steps': 2,
        'num_epochs_per_network': {
            4: 10,
            8: 20,
            16: 20,
            32: 30,
            64: 40,
            128: 80
        },
        'transition_length_per_network': {
            8: 25000,
            16: 25000,
            32: 37500,
            64: 50000,
            128: 100000
        },
        'epoch_length_per_network': {
            4: 2500,
            8: 2500,
            16: 2500,
            32: 2500,
            64: 2500,
            128: 2500
        }
    },
    '4x4_generation': {
        'network_type': 'network4',
        'data_dir_per_network_size': {
            4: '/home/datasets/celeba-aligned/4x4',
            8: '/home/datasets/celeba-aligned/8x8',
            16: '/home/datasets/celeba-aligned/16x16',
            32: '/home/datasets/celeba-aligned/32x32',
            64: '/home/datasets/celeba-aligned/64x64',
            128: '/home/datasets/celeba-aligned/128x128'
        },
        'dry_run': False,
        'save_real_images': True,
        'model_save_frequency': 4,
        'image_save_frequency':200,
        'training_set_size': 10000,
        'gradient_penalty_factor': 10,
        'learning_rate': 1e-3,
        'mini_batch_size': 256,
        'num_critic_training_steps': 2,
        'num_epochs_per_network': {
            4: 10,
            8: 20,
            16: 20,
            32: 30,
            64: 40,
            128: 80
        },
        'transition_length_per_network': {
            8: 25000,
            16: 25000,
            32: 37500,
            64: 50000,
            128: 100000
        },
        'epoch_length_per_network': {
            4: 1000,
            8: 2500,
            16: 2500,
            32: 2500,
            64: 2500,
            128: 2500
        }
    },
    'test_flipping_images': {
        'network_type': 'network',
        'data_dir_per_network_size': {
            4: '/home/datasets/celeba-aligned/4x4',
            8: '/home/datasets/celeba-aligned/8x8',
            16: '/home/datasets/celeba-aligned/16x16',
            32: '/home/datasets/celeba-aligned/32x32',
            64: '/home/datasets/celeba-aligned/64x64',
            128: '/home/datasets/celeba-aligned/128x128'
        },
        'dry_run': True,
        'model_save_frequency': 4,
        'image_save_frequency': 2,
        'training_set_size': 10,
        'gradient_penalty_factor': 10,
        'learning_rate': 1e-4,
        'mini_batch_size': 2,
        'num_critic_training_steps': 1,
        'num_epochs_per_network': {
            4: 1,
            8: 1,
            16: 1,
            32: 1,
            64: 1,
            128: 1
        },
        'transition_length_per_network': {
            8: 2,
            16: 2,
            32: 2,
            64: 2,
            128: 2
        },
        'epoch_length_per_network': {
            4: 1,
            8: 1,
            16: 1,
            32: 1,
            64: 1,
            128: 1
        }
    },
    'test_saving_data': {
        'network_type': 'network',
        'data_dir_per_network_size': {
            4: '/home/datasets/celeba-aligned/4x4',
            8: '/home/datasets/celeba-aligned/8x8',
            16: '/home/datasets/celeba-aligned/16x16',
            32: '/home/datasets/celeba-aligned/32x32',
            64: '/home/datasets/celeba-aligned/64x64',
            128: '/home/datasets/celeba-aligned/128x128'
        },
        'dry_run': False,  # Test if data is saved correctly.
        'model_save_frequency': 4,
        'image_save_frequency': 2,
        'training_set_size': 10,
        'gradient_penalty_factor': 10,
        'learning_rate': 1e-4,
        'mini_batch_size': 2,
        'num_critic_training_steps': 2,
        'num_epochs_per_network': {
            4: 4,
            8: 4,
            16: 4,
            32: 4,
            64: 4,
            128: 4
        },
        'transition_length_per_network': {
            8: 4,
            16: 4,
            32: 4,
            64: 4,
            128: 4
        },
        'epoch_length_per_network': {
            4: 2,
            8: 2,
            16: 2,
            32: 2,
            64: 2,
            128: 2
        }
    },
    'test_dry_run': {
        'network_type': 'network',
        'data_dir_per_network_size': {
            4: '/home/datasets/celeba-aligned/4x4',
            8: '/home/datasets/celeba-aligned/8x8',
            16: '/home/datasets/celeba-aligned/16x16',
            32: '/home/datasets/celeba-aligned/32x32',
            64: '/home/datasets/celeba-aligned/64x64',
            128: '/home/datasets/celeba-aligned/128x128'
        },
        'dry_run': True,  # No data is saved.
        'model_save_frequency': 4,
        'image_save_frequency': 2,
        'training_set_size': 10,
        'gradient_penalty_factor': 10,
        'learning_rate': 1e-4,
        'mini_batch_size': 2,
        'num_critic_training_steps': 2,
        'num_epochs_per_network': {
            4: 4,
            8: 4,
            16: 4,
            32: 4,
            64: 4,
            128: 4
        },
        'transition_length_per_network': {
            8: 4,
            16: 4,
            32: 4,
            64: 4,
            128: 4
        },
        'epoch_length_per_network': {
            4: 2,
            8: 2,
            16: 2,
            32: 2,
            64: 2,
            128: 2
        }
    }
}

def get_configuration(name):
    return Configuration(_CONFIGURATIONS[name])


class Configuration():
    
    def __init__(self, configuration):
        self.configuration = configuration
        self.required_fields = _REQUIRED_FIELDS

    def to_dictionary(self):
        return self.configuration

    def is_enabled(self, field):
        return field in self.configuration and self.configuration[field]

    def is_disabled(self, field):
        return not self.is_enabled(field)

    def get(self, field, default=None):
        return self.configuration[field] if self.is_enabled(field) else default

    def missing_fields(self):
        missing_fields = []

        for required_field in self.required_fields:
            if required_field not in self.configuration:
                missing_fields.append(required_field)

        return missing_fields
