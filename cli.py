import argparse
from model.process import ModelProcessor


def main():
    parser = argparse.ArgumentParser(description="NFL Game Prediction CLI")
    parser.add_argument(
        "--process",
        action="store_true",
        help="Process input data and generate model matrices",
    )

    args = parser.parse_args()

    if args.process:
        processor = ModelProcessor()
        processor.process_and_save()
        print("Processed data and saved train/validation sets")


if __name__ == "__main__":
    main()
