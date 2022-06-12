import gan.config.hyperparameters as hyperparameters
import torch
import torch.nn as nn
import torch.optim as optim


class Checkpoint:
    @classmethod
    def load(cls, model: nn.Module, optimizer: optim.Optimizer, input_filepath: str) -> None:
        print('=> Loading checkpoint')

        try:
            checkpoint = torch.load(input_filepath, map_location=hyperparameters.DEVICE)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            for param_group in optimizer.param_groups:
                param_group['lr'] = hyperparameters.LEARNING_RATE

        except FileNotFoundError:
            print('There is no file to load data from.')

    @classmethod
    def save(cls, model: nn.Module, optimizer: optim.Optimizer, output_filepath: str) -> None:
        print('=> Saving checkpoint')

        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, output_filepath)
