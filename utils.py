import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.nn import Module as NNModule

from exceptions import ImageTrainerError


def get_last_child_module(model: NNModule) -> Tuple[str, NNModule]:
    return list(model.named_children())[-1]


def get_device(gpu: bool = False) -> torch.device:
    """Get the torch device, try to get a GPU if gpu is set, raise an error if no GPU and gpu is set"""
    if gpu:
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            raise ImageTrainerError(
                f"No GPU available. Run without -g/--gpu argument"
            )
    torch.device("cpu")


def parse_json_file(cat_name_path: Path) -> Dict:
    return json.loads(cat_name_path.read_text())


def invert_dict(the_dict: Dict) -> Dict:
    return {v: k for k, v in the_dict.items()}
