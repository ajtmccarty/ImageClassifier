from pathlib import Path
from typing import List

from torch.nn import Module as NNModule

from cli_parser import create_training_parser


class ImageTrainer:
    def __init__(
            self,
            data_dir: Path,
            save_dir: Path,
            arch: NNModule,
            learning_rate: float,
            hidden_units: List[int],
            epochs: int,
            gpu: bool
    ):
        # initial instance vars
        self.data_dir: Path = data_dir
        self.save_dir: Path = save_dir
        self.arch: NNModule = arch
        self.learning_rate: float = learning_rate
        self.hidden_units: List[int] = hidden_units
        self.epochs: int = epochs
        self.gpu: bool = gpu


if __name__ == "__main__":
    parser = create_training_parser()
    kwargs = vars(parser.parse_args())
    print(kwargs)
    img_trainer = ImageTrainer(**kwargs)
