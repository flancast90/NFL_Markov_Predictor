import argparse
from model.process import ModelProcessor
from model.train import ModelTrainer
from model.validate import ModelValidator


def main():
    parser = argparse.ArgumentParser(description="NFL Game Prediction CLI")
    parser.add_argument(
        "--process",
        action="store_true",
        help="Process input data and generate model matrices",
    )
    parser.add_argument(
        "--train", action="store_true", help="Train the HMM model on processed data"
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate the trained model"
    )

    args = parser.parse_args()

    if args.process:
        processor = ModelProcessor()
        processor.process_and_save()
        print("Processed data and saved train/validation sets")

    if args.train:
        trainer = ModelTrainer()
        trainer.load_training_data()
        trainer.train()
        trainer.save_model()
        print("Trained and saved model")

    if args.validate:
        validator = ModelValidator()
        validator.load_model()
        validator.load_validation_data()
        accuracy, confusion_matrix = validator.validate()
        print(f"Model validation complete. Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
