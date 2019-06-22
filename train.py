import re
from pathlib import Path
from time import time
from typing import Dict, List, Tuple

import torch
from torch.optim import Adam, Optimizer
from torch.nn import (
    Dropout,
    Linear,
    LogSoftmax,
    Module as NNModule,
    NLLLoss,
    ReLU,
    Sequential,
)
from torch.utils.data import DataLoader
from torchvision import transforms, models as torch_models
from torchvision.datasets import DatasetFolder, ImageFolder


from cli_parser import create_training_parser
from workspace_utils import keep_awake


class ImageTrainerError(Exception):
    pass


class ImageTrainerInitError(ImageTrainerError):
    pass


def get_last_child_module(model: NNModule) -> Tuple[str, NNModule]:
    return list(model.named_children())[-1]


class ImageTrainer:
    def __init__(
        self,
        data_dir: Path,
        save_dir: Path,
        arch: str,
        learning_rate: float,
        hidden_units: List[int],
        epochs: int,
        gpu: bool,
    ):
        # initial instance vars
        self.data_dir: Path = data_dir
        self.save_dir: Path = save_dir
        self.learning_rate: float = learning_rate
        self.hidden_units: List[int] = hidden_units
        self.epochs: int = epochs
        self.optimizer_class: Optimizer = Adam
        self.criterion: NNModule = NLLLoss()

        # TODO: this verification should probably be in the cli_parser
        if gpu:
            if torch.cuda.is_available():
                self.gpu: bool = True
                self.device: torch.device = torch.device("cuda")
            else:
                raise ImageTrainerInitError(
                    f"No GPU available. Run without -g/--gpu argument"
                )
        else:
            self.gpu = False
            self.device = torch.device("cpu")

        self.class_to_idx: Dict[str, int] = dict()
        self.dataloaders: Dict[str, DataLoader] = dict()
        self.generate_dataloaders(self.data_dir)

        # download the model last because it takes a long time we want to be sure the rest
        # of the initialization was successful so users aren't waiting around for errors
        self.arch: NNModule = self.initialize_pretrained_model(arch)

        classifier_in_nodes: int = self.get_classifier_input_size(self.arch)
        classifier_out_nodes: int = self.get_num_cats(Path(self.data_dir, "test"))
        layer_sizes: List[int] = [
            classifier_in_nodes,
            *self.hidden_units,
            classifier_out_nodes,
        ]
        self.classifier = self.build_classifier(layer_sizes)

        self.last_layer_name: str = get_last_child_module(self.arch)[0]
        setattr(self.arch, self.last_layer_name, self.classifier)

    @staticmethod
    def initialize_pretrained_model(model_name: str) -> NNModule:
        """Get the torchvision model by name, turn off gradient descent of its features, add a name instance var"""
        model_class: NNModule = getattr(torch_models, model_name)
        vision_model: NNModule = model_class(pretrained=True)
        for p in vision_model.parameters():
            p.requires_grad = False
        vision_model.name = model_name
        return vision_model

    @staticmethod
    def get_classifier_input_size(model: NNModule) -> int:
        """The torchvision models either end with a classifier or an "fc" layer"""
        # if the classifier is just a single layer
        last_layer: NNModule = get_last_child_module(model)[1]
        if hasattr(last_layer, "in_features"):
            return last_layer.in_features
        for module in last_layer:
            try:
                return module.in_features
            except AttributeError:
                pass
        raise ImageTrainerError(
            f"Cannot determine classifier input width. Model "
            f"must have a classifier with in_features attribute "
            f"or an 'fc' layer with an in_features attribute"
        )

    @staticmethod
    def get_num_cats(image_dir: Path) -> int:
        """Return the number of directories in one of the image directories, which will be the number
        of categories"""
        return len(list(image_dir.iterdir()))

    @staticmethod
    def build_classifier(layers: List[int], dropout: float = 0.3) -> Sequential:
        args: List[NNModule] = []
        for i in range(len(layers) - 2):
            from_size = layers[i]
            to_size = layers[i + 1]
            args.extend([Linear(from_size, to_size), ReLU(), Dropout(dropout)])
        args.extend([Linear(layers[-2], layers[-1]), LogSoftmax(dim=1)])
        classifier: Sequential = Sequential(*args)
        classifier.dropout = dropout
        return classifier

    @staticmethod
    def load_model(chkpnt_path: Path, dropout: float = 0.3):
        # load it to the CPU by default b/c that always exists
        data: Dict = torch.load(str(chkpnt_path), map_location="cpu")
        name: str = data["name"]
        last_layer_name: str = data["last_layer_name"]
        model = ImageTrainer.initialize_pretrained_model(name)

        dropout = float(data["dropout"]) if "dropout" in data else dropout
        sizes: List[int] = []
        first: bool = True
        for k, v in data["model_state"].items():
            if re.match(r"{}\.[0-9]*\.weight".format(last_layer_name), k):
                if first:
                    first = False
                    sizes.append(v.shape[1])
                sizes.append(v.shape[0])
        classifier: NNModule = ImageTrainer.build_classifier(sizes, dropout)
        setattr(model, last_layer_name, classifier)
        model.load_state_dict(data["model_state"])
        model.class_to_idx = data["class_to_idx"]
        return model

    def save_model(self) -> Path:
        """Save the model in the specified directory"""
        checkpoint = {
            "dropout": self.arch.classifier.dropout,
            "name": self.arch.name,
            "class_to_idx": self.class_to_idx,
            "model_state": self.arch.state_dict(),
            "last_layer_name": self.last_layer_name,
        }
        timestamp = str(int(time()))
        name: str = self.arch.name
        # make sure the directory exists
        self.save_dir.mkdir(exist_ok=True)
        chkpnt_path: Path = Path(self.save_dir, f"checkpoint-{name}_{timestamp}.torch")
        torch.save(checkpoint, str(chkpnt_path))
        return chkpnt_path

    def generate_dataloaders(self, data_dir: Path, batch_size: int = 32) -> None:
        """Initialize `dataloaders` and `class_to_idx` instance attr"""
        image_dirs: Dict[str, Path] = {
            "train": Path(data_dir, "train").absolute(),
            "validation": Path(data_dir, "valid").absolute(),
            "test": Path(data_dir, "test").absolute(),
        }
        # validate image dirs
        for name, p in image_dirs.items():
            if not (p.exists() and p.is_dir()):
                raise ImageTrainerError(
                    f"Path {p} for {name} image directory is not a valid directory"
                )

        the_transforms: Dict[str, transforms.Compose] = {
            "train": transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(45),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            # validation is same as test for now
            "validation": transforms.Compose(
                [
                    transforms.Resize(255),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            "test": transforms.Compose(
                [
                    transforms.Resize(255),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
        }

        image_datasets: Dict[str, DatasetFolder] = {
            name: ImageFolder(str(image_dirs[name]), transform=the_transforms[name])
            for name in image_dirs.keys()
        }
        self.class_to_idx = image_datasets["train"].class_to_idx

        self.dataloaders = {
            name: DataLoader(image_datasets[name], batch_size=batch_size, shuffle=True)
            for name in ["train", "validation", "test"]
        }

    def train_and_validate(self):
        optimizer: NNModule = self.optimizer_class(
            self.classifier.parameters(), lr=self.learning_rate
        )
        self.arch.to(self.device)
        for ep in keep_awake(range(self.epochs)):
            print(f"\nStarting epoch # {ep + 1} of {self.epochs}")
            print(f"Batch progress", end="...")
            # set model for training
            self.arch.train()
            training_loss: float = 0.0

            for count, (images, labels) in enumerate(self.dataloaders["train"]):
                optimizer.zero_grad()
                images: torch.Tensor = images.to(self.device)
                labels: torch.Tensor = labels.to(self.device)

                if count % 10 == 0:
                    print(count, end="...")
                log_ps: torch.Tensor = self.arch.forward(images)
                loss: torch.Tensor = self.criterion(log_ps, labels)
                training_loss += loss.item()
                loss.backward()
                optimizer.step()
            print(f"\nTotal training loss: {training_loss}")

            print(f"\nBeginning evaluation for epoch #{ep + 1}")
            print(f"Batch progress", end="...")
            # set model for evaluation
            self.arch.eval()
            accuracy: float = 0.0
            validation_loss: float = 0.0
            with torch.no_grad():

                for count, (images, labels) in enumerate(
                    self.dataloaders["validation"]
                ):
                    images: torch.Tensor = images.to(self.device)
                    labels: torch.Tensor = labels.to(self.device)

                    if count % 10 == 0:
                        print(count, end="...")
                    log_ps: torch.Tensor = self.arch.forward(images)
                    loss: torch.Tensor = self.criterion(log_ps, labels)
                    validation_loss += loss.item()
                    ps: torch.Tensor = torch.exp(log_ps)
                    top_class: torch.Tensor = ps.topk(1, dim=1)[1]
                    equals: torch.Tensor = torch.eq(
                        top_class, labels.view(*top_class.shape)
                    )
                    batch_acc: float = equals.type(torch.FloatTensor).mean()
                    accuracy += batch_acc

                print(
                    f"\n\tTotal validation loss: {validation_loss}"
                    f"\n\tAccuracy: {accuracy / len(self.dataloaders['validation'])}"
                )

    def test_model(self):
        print("\nStarting evaluation on test data...")
        print("Batch progress", end="...")
        self.arch.to(self.device)
        self.arch.eval()
        accuracy: float = 0.0
        # turn off gradient descent for testing
        with torch.no_grad():
            for count, (images, labels) in enumerate(self.dataloaders["test"]):
                images: torch.Tensor = images.to(self.device)
                labels: torch.Tensor = labels.to(self.device)

                if count % 10 == 0:
                    print(count, end="...")
                log_ps = self.arch.forward(images)
                ps: torch.Tensor = torch.exp(log_ps)
                top_class: torch.Tensor = ps.topk(1, dim=1)[1]
                equals = torch.eq(top_class, labels.view(*top_class.shape))
                batch_acc: float = equals.type(torch.FloatTensor).mean()
                accuracy += batch_acc

        print(f"\nTest Accuracy: {accuracy/len(self.dataloaders['test'])}")


if __name__ == "__main__":
    parser = create_training_parser()
    kwargs = vars(parser.parse_args())
    img_trainer = ImageTrainer(**kwargs)
    img_trainer.train_and_validate()
    img_trainer.test_model()
    chkpnt: Path = img_trainer.save_model()
    print(f"Saved checkpoint to {chkpnt}")
    model = ImageTrainer.load_model(chkpnt)
    print(model)
