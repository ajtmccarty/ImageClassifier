from pathlib import Path

from cli_parser import create_training_parser
from image_trainer import ImageTrainer

if __name__ == "__main__":
    parser = create_training_parser()
    kwargs = vars(parser.parse_args())
    img_trainer = ImageTrainer(**kwargs)
    img_trainer.train_and_validate()
    img_trainer.test_model()
    chkpnt: Path = img_trainer.save_model()
    print(f"Saved checkpoint to {chkpnt}")
