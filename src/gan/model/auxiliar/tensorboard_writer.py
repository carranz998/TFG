
from torch.utils.tensorboard import SummaryWriter

class TensorBoardWriter:
    def __init__(self) -> None:
        self.writer = SummaryWriter(f'logs/gan1')