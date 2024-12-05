import argparse
from model.process import ModelProcessor
from model.train import ModelTrainer
from model.validate import ModelValidator
from montecarlo import MonteCarloSimulator
import algs

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
    parser.add_argument(
        "--dataset", help="Select the desired dataset (Default dataset is data/nfl_dataset.json)"
    )
    parser.add_argument(
        "--model", help="Select the name of the model (Will save at model/saves/model_*model*.json, defaults to model/saves/trained_model.json)"
    )
    parser.add_argument(
        "--montecarlo", type=int, metavar='N', help="Run a Monte Carlo simulation on the given training data with N iterations"
    )


    args = parser.parse_args()

    if args.dataset is not None:
        print("Dataset located at " + args.dataset + " loaded")

    if args.process:
        """Process the given data model into a training set and validation set"""
        processor = ModelProcessor()
        if args.dataset is not None:
            processor.process_and_save(args.dataset)
        else:
            processor.process_and_save()
        print("Processed data and saved train/validation sets")

    if args.train:
        """Train a new data model using the training set"""
        trainer = ModelTrainer()
        trainer.load_training_data()
        trainer.train()
        if args.model is not None:
            trainer.save_model("model/saves/model_" + args.model + ".json")
        else:
            trainer.save_model()
        print("Trained and saved model")

    if args.validate:
        """Validate the selected data model using validation set"""
        validator = ModelValidator()
        if args.model is not None:
            validator.load_model("model/saves/model_" + args.model + ".json")
        else:
            validator.load_model()
        validator.load_validation_data()
        accuracy, confusion_matrix, total_profit, roi = validator.validate()
        print(f"Model validation complete. Accuracy: {accuracy:.4f}")
    if args.montecarlo:
        validator = ModelValidator()
        print("Loading model...")
        validator.load_model()
        print("Loading validation data...")
        validator.load_validation_data()
        print(f"Validation data loaded: {len(validator.validation_data) if validator.validation_data else 'None'} games")
        
        simulator = MonteCarloSimulator(validator, num_simulations=args.montecarlo)
        results = simulator.simulate()
        simulator.save_results(results)
        print(f"Monte Carlo simulation with {args.montecarlo} iterations complete. Results saved to file.")


if __name__ == "__main__":
    main()
