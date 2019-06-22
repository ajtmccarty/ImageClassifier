from cli_parser import create_prediction_parser
from image_trainer import ImagePredictor


if __name__ == "__main__":
    parser = create_prediction_parser()
    kwargs = vars(parser.parse_args())
    predictor = ImagePredictor(**kwargs)
    predictor.handle_prediction()
