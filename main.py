"""
Orquestrador
This file allows to run the following commands on terminal:
python main.py preprocess
python main.py train
python main.py evaluate
python main.py predict --file data/raw/video123.pkl

"""

import argparse
from src.preprocessing.build_dataset import run_preprocessing
from src.training.train import run_training
from src.training.evaluate import run_evaluation
from src.inference.predict import run_prediction

def main():
    parser = argparse.ArgumentParser(description="UCF101 Skeleton Pipeline")
    subparsers = parser.add_subparsers(dest="command")

    # --- PREPROCESS ---
    preprocess = subparsers.add_parser("preprocess", help="Preprocess raw skeleton data")
    
    # --- TRAIN ---
    train = subparsers.add_parser("train", help="Train model")
    
    # --- EVALUATE ---
    evaluate = subparsers.add_parser("evaluate", help="Evaluate model")
    
    # --- PREDICT ---
    predict = subparsers.add_parser("predict", help="Predict on single skeleton file")
    predict.add_argument("--file", required=True, help="Path to skeleton .pkl")

    args = parser.parse_args()

    if args.command == "preprocess":
        run_preprocessing()

    elif args.command == "train":
        run_training()

    elif args.command == "evaluate":
        run_evaluation()

    elif args.command == "predict":
        run_prediction(args.file)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
