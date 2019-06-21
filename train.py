from argparse import ArgumentParser, ArgumentTypeError

import torchvision.models


def int_list(in_string):
    """Verify in_string is comma-separated integers"""
    try:
        return [int(x) for x in in_string.split(",")]
    except ValueError as exc:
        raise ArgumentTypeError(f"{in_string} must be a comma-separated list of integers") from exc

def torchvision_module(in_string):
    """Verify that in_string is the name of a model in torchvision.models"""
    if hasattr(torchvision.models, in_string):
        return in_string
    raise ArgumentTypeError(f"{in_string} is not a torchvision model")

def create_training_parser():
    parser = ArgumentParser(
        prog="train",
        description="Train a neural net classifier for a pretrained model from PyTorch's torchvision module"
    )
    parser.add_argument(
        "data-dir",
        type=str,
        help="Directory of the structure specified by PyTorch's ImageFolder class(https://pytorch.org"
        "/docs/0.4.0/torchvision/datasets.html?highlight=imagefolder#imagefolder"
    )
    parser.add_argument(
        "-s",
        "--save-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints of trained neural nets"
    )
    parser.add_argument(
        "-a",
        "--arch",
        type=torchvision_module,
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
        type=int_list,
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

if __name__ == "__main__":
    parser = create_training_parser()
    kwargs = vars(parser.parse_args())
    print(kwargs)
    