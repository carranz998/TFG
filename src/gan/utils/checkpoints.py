import torch
import torch.nn as nn


class Checkpoint:
    @classmethod
    def load(cls, model: nn.Module, optimizer: torch.optim.Optimizer, input_filename: str, lr: float) -> None:
        print('=> Loading checkpoint')

        checkpoint = torch.load(input_filename, map_location='cuda')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    @classmethod
    def save(cls, model: nn.Module, optimizer: torch.optim.Optimizer, output_filename: str) -> None:
        print('=> Saving checkpoint')

        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, output_filename)
