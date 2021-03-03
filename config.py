'''Defines configurations for experiments.

Defines different configurations with custom parameters to run experiments.
Configurations can be specified as a command line argument.
'''

_REQUIRED_FIELDS = [
    'data_dir',
    'model_save_frequency',
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
        'data_dir': '/home/datasets/celeba-aligned',
        'dry_run': False,
        'model_save_frequency': 4,
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
    'test_saving_data': {
        'data_dir': '/home/datasets/celeba-aligned',
        'dry_run': False,  # Test if data is saved correctly.
        'model_save_frequency': 4,
        'training_set_size': 10,
        'gradient_penalty_factor': 10,
        'learning_rate': 1e-4,
        'mini_batch_size': 2,
        'num_critic_training_steps': 2,
        'num_epochs_per_network': {
            4: 10,
            8: 20,
            16: 20,
            32: 20,
            64: 20,
            128: 20
        },
        'transition_length_per_network': {
            8: 2,
            16: 2,
            32: 2,
            64: 2,
            128: 2
        },
        'epoch_length_per_network': {
            4: 4,
            8: 4,
            16: 4,
            32: 4,
            64: 4,
            128: 4
        }
    },
    'test_dry_run': {
        'data_dir': '/home/datasets/celeba-aligned',
        'dry_run': True,  # No data is saved.
        'model_save_frequency': 4,
        'training_set_size': 10,
        'gradient_penalty_factor': 10,
        'learning_rate': 1e-4,
        'mini_batch_size': 2,
        'num_critic_training_steps': 2,
        'num_epochs_per_network': {
            4: 10,
            8: 20,
            16: 20,
            32: 20,
            64: 20,
            128: 20
        },
        'transition_length_per_network': {
            8: 2,
            16: 2,
            32: 2,
            64: 2,
            128: 2
        },
        'epoch_length_per_network': {
            4: 4,
            8: 4,
            16: 4,
            32: 4,
            64: 4,
            128: 4
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
        has_required_fields = True
        missing_fields = []

        for required_field in self.required_fields:
            if required_field not in self.configuration:
                missing_fields.append(required_field)
                has_required_fields = False

        return None if has_required_fields else missing_fields
