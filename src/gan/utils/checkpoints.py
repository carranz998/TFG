import os

import gan.config.hyperparameters as hyperparameters
import torch
import torch.nn as nn


class Checkpoint:
    @classmethod
    def load(cls, model: nn.Module, optimizer: torch.optim.Optimizer, input_filepath: str) -> None:
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
    def save(cls, model: nn.Module, optimizer: torch.optim.Optimizer, output_filepath: str) -> None:
        print('=> Saving checkpoint')
        
        if not os.path.exists(output_filepath):
            cls.__create_file(output_filepath)

        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, output_filepath)

    @classmethod
    def __create_file(cls, filepath: str) -> None:
        dirname = os.path.dirname(filepath)
        os.makedirs(dirname, exist_ok=True)

        with open(filepath, 'w'): 
            pass
