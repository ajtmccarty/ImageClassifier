from cli_parser import create_training_parser


class ImageTrainer:
    def __init__(self, data_dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu):
        self.data_dir: Path = data_dir
        self.save_dir = save_dir
        self.arch = arch
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units
        self.epochs = epochs
    

if __name__ == "__main__":
    parser = create_training_parser()
    kwargs = vars(parser.parse_args())
    print(kwargs)
    img_trainer = ImageTrainer(**kwargs)
    