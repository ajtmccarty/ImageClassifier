from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path
from typing import List

from torch.nn import Module as NNModule
import torchvision.models


class ArgTypes:
    """Class to organize different types of arguments"""
    
    @staticmethod
    def int_list(in_string: str) -> List[int]:
        """Verify in_string is comma-separated integers"""
        try:
            return [int(x) for x in in_string.split(",")]
        except ValueError as exc:
            raise ArgumentTypeError(f"{in_string} must be a comma-separated list of integers") from exc

    @staticmethod
    def torchvision_model(in_string: str) -> NNModule:
        """Verify that in_string is the name of a model in torchvision.models"""
        if hasattr(torchvision.models, in_string):
            return getattr(torchvision.models, in_string)
        raise ArgumentTypeError(f"{in_string} is not a torchvision model")

    @staticmethod
    def valid_dir(in_string: str) -> Path:
        """Verify that in string is a valid directory path and convert it to an absolute Path"""
        p: Path = Path(in_string).absolute()
        if not (p.exists() and p.is_dir()):
            raise ArgumentTypeError(f"{in_string} is not a valid path.")
        return p

    @staticmethod
    def abs_path(in_string: str) -> Path:
        """Turn a string into an absolute path"""
        return Path(in_string).absolute()


def create_training_parser():
    parser = ArgumentParser(
        prog="train",
        description="Train a neural net classifier for a pretrained model from PyTorch's torchvision module"
    )
    parser.add_argument(
        "data_dir",
        type=ArgTypes.valid_dir,
        help="Directory of the structure specified by PyTorch's ImageFolder class(https://pytorch.org"
        "/docs/0.4.0/torchvision/datasets.html?highlight=imagefolder#imagefolder"
    )
    parser.add_argument(
        "-s",
        "--save-dir",
        type=ArgTypes.abs_path,
        default="checkpoints",
        help="Directory to save checkpoints of trained neural nets"
    )
    parser.add_argument(
        "-a",
        "--arch",
        type=ArgTypes.torchvision_model,
        default="vgg16",
        help="Name of pretrained neural net to use. Must be one of the models in PyTorch's torchvision.models("
             "https://pytorch.org/docs/0.4.0/torchvision/models.html), such as vgg16, densenet121, etc." 
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=0.003,
        help="Learning rate for the model being trained"
    )
    parser.add_argument(
        "-hu",
        "--hidden-units",
        type=ArgTypes.int_list,
        default="500,400",
        help="Comma-separated list of ints representing the size of each hidden layer. A single layer (i.e. 400) is allowed."
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to train the model for"
    )
    parser.add_argument(
        "-g",
        "--gpu",
        action="store_true",
        help="Whether to use a GPU. Will only work if PyTorch has access to a GPU"
    )
    return parser