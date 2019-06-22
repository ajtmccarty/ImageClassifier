from pathlib import Path
from typing import Dict, List

import torch
from torch.nn import Module as NNModule
from torch.utils.data import DataLoader
from torchvision import transforms, models as torch_models
from torchvision.datasets import DatasetFolder, ImageFolder


from cli_parser import create_training_parser


class ImageTrainerError(Exception):
    pass


class ImageTrainerInitError(ImageTrainerError):
    pass


class ImageTrainer:
    def __init__(
            self,
            data_dir: Path,
            save_dir: Path,
            arch: str,
            learning_rate: float,
            hidden_units: List[int],
            epochs: int,
            gpu: bool
    ):
        # initial instance vars
        self.data_dir: Path = data_dir
        self.save_dir: Path = save_dir
        self.learning_rate: float = learning_rate
        self.hidden_units: List[int] = hidden_units
        self.epochs: int = epochs

        # TODO: this verification should probably be in the cli_parser
        if gpu:
            if torch.cuda.is_available():
                self.gpu: bool = True
                self.device: torch.device = torch.device("cuda")
            else:
                raise ImageTrainerInitError(f"No GPU available. Run without -g/--gpu argument")
        else:
            self.gpu = False
            self.device = torch.device("cpu")

        self.dataloaders: Dict[str, DataLoader] = self.generate_dataloaders(self.data_dir)

        # download the model last because it takes a long time we want to be sure the rest
        # of the initialization was successful so users aren't waiting around for errors
        self.arch: NNModule = self.initialize_pretrained_model(arch)

    @staticmethod
    def generate_dataloaders(data_dir: Path, batch_size: int = 32) -> Dict[str, DataLoader]:
        """Initialize `dataloaders` instance attr"""
        image_dirs: Dict[str, Path] = {
            "train": Path(data_dir, 'train').absolute(),
            "validation": Path(data_dir, 'valid').absolute(),
            "test":  Path(data_dir, 'test').absolute()
        }
        # validate image dirs
        for name, p in image_dirs.items():
            if not (p.exists() and p.is_dir()):
                raise ImageTrainerError(f"Path {p} for {name} image directory is not a valid directory")

        the_transforms: Dict[str, transforms.Compose] = {
            "train": transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(45),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            # validation is same as test for now
            "validation": transforms.Compose([
                transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            "test": transforms.Compose([
                transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        image_datasets: Dict[str, DatasetFolder] = {
            name: ImageFolder(str(image_dirs[name]), transform=the_transforms[name])
            for name in image_dirs.keys()
        }

        return {
            name: DataLoader(image_datasets[name], batch_size=batch_size, shuffle=True)
            for name in ["train", "validation", "test"]
        }

    @staticmethod
    def initialize_pretrained_model(model_name: str) -> NNModule:
        """Get the torchvision model by name, turn off gradient descent of its features, add a name instance var"""
        model_class: NNModule = getattr(torch_models, model_name)
        vision_model: NNModule = model_class(pretrained=True)
        for p in vision_model.parameters():
            p.requires_grad = False
        vision_model.name = model_name
        return vision_model

    torch_models.resnet18()

    @staticmethod
    def get_classifier_input_size(model: NNModule):
        """The torchvision models either end with a classifier or an "fc" layer"""
        # if the classifier is just a single layer
        if hasattr(model, "fc"):
            return model.fc.in_features
        if hasattr(model.classifier, "in_features"):
            return model.classifier.in_features
        for layer in model.classifier:
            try:
                return layer.in_features
            except AttributeError:
                pass
        raise ImageTrainerError(f"Cannot determine classifier input width. Model "
                                f"must have a classifier with in_features attribute "
                                f"or an 'fc' layer with an in_features attribute")


if __name__ == "__main__":
    parser = create_training_parser()
    kwargs = vars(parser.parse_args())
    print(kwargs)
    img_trainer = ImageTrainer(**kwargs)
    print(img_trainer.dataloaders)
    print(img_trainer.arch)
    print(img_trainer.device)
