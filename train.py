from pathlib import Path
from typing import Dict, List

from torch.nn import Module as NNModule
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import DatasetFolder, ImageFolder


from cli_parser import create_training_parser


class ImageTrainerError(Exception):
    pass


class ImageTrainerInitErro(ImageTrainerError):
    pass


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

        self.dataloaders: Dict[str, DataLoader] = {}

    def generate_dataloaders(self, batch_size: int = 32):
        """Initialize `dataloaders` instance attr"""
        image_dirs: Dict[str, Path] = {
            "train": Path(self.data_dir, 'train').absolute(),
            "validation": Path(self.data_dir, 'valid').absolute(),
            "test":  Path(self.data_dir, 'test').absolute()
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

        self.dataloaders = {
            name: DataLoader(image_datasets[name], batch_size=batch_size, shuffle=True)
            for name in ["train", "validation", "test"]
        }


if __name__ == "__main__":
    parser = create_training_parser()
    kwargs = vars(parser.parse_args())
    print(kwargs)
    img_trainer = ImageTrainer(**kwargs)
