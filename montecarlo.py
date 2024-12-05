from typing import List, Dict, Tuple, Optional
import numpy as np
from datetime import datetime
from model.validate import ModelValidator
from dataclasses import dataclass
import pandas as pd


@dataclass
class SimulationPath:
    accuracy: List[float]
    profit: List[float]
    roi: List[float]
    games: List[Dict]


class MonteCarloSimulator:
    def __init__(self, validator: ModelValidator, num_simulations: int = 1000):
        """
        Initialize Monte Carlo simulator

        Args:
            validator: ModelValidator instance with loaded model and validation data
            num_simulations: Number of simulations to run
        """
        self.validator = validator
        self.num_simulations = num_simulations
        self.results = []

    def create_bootstrap_sample(self, data: List[Dict]) -> List[Dict]:
        """
        Create a bootstrapped dataset by sampling with replacement.
        Should maintain temporal ordering for historical calculations.

        Args:
            data: List of game dictionaries
        Returns:
            Bootstrapped sample of games
        """
        # Sort data by date to maintain temporal ordering
        sorted_data = sorted(data, key=lambda x: x["date"])

        # Sample with replacement
        n = 50
        indices = np.random.choice(len(sorted_data), size=n, replace=True)

        # Create bootstrapped sample while preserving temporal order
        bootstrap_sample = [sorted_data[i] for i in sorted(indices)]

        return bootstrap_sample

    def run_simulation(
        self, validation_data: List[Dict]
    ) -> Tuple[float, float, float, SimulationPath]:
        """
        Run a single simulation with bootstrapped data

        Args:
            validation_data: List of validation game dictionaries
        Returns:
            Tuple of (accuracy, total_profit, roi, simulation_path)
        """
        # Create bootstrap sample
        bootstrap_data = self.create_bootstrap_sample(validation_data)

        # Update validator with bootstrapped data
        self.validator.validation_data = bootstrap_data

        # Ensure model is loaded
        if not hasattr(self.validator, "model") or self.validator.model is None:
            self.validator.load_model()

        # Track metrics over time
        running_accuracy = []
        running_profit = []
        running_roi = []

        correct = 0
        total = 0
        profit = 0

        for i, game in enumerate(bootstrap_data, 1):
            prediction = self.validator.predict(game["home_team"], game["away_team"])
            actual = (
                "home_win"
                if int(game["home_score"]) > int(game["away_score"])
                else "away_win"
            )

            correct += prediction == actual
            total += 1

            # Calculate running metrics
            current_accuracy = correct / total
            running_accuracy.append(current_accuracy)

            # Simplified profit calculation
            profit += 1 if prediction == actual else -1
            running_profit.append(profit)

            # Calculate ROI
            roi = (profit / total) * 100
            running_roi.append(roi)

        path = SimulationPath(
            accuracy=running_accuracy,
            profit=running_profit,
            roi=running_roi,
            games=bootstrap_data,
        )

        return current_accuracy, profit, roi, path

    def simulate(self, store_paths: bool = False) -> Dict:
        """
        Run multiple simulations and analyze results

        Args:
            store_paths: Whether to store and return individual simulation paths
        Returns:
            Dictionary containing statistical results and optional paths
        """
        validation_data = self.validator.validation_data
        accuracies = []
        profits = []
        rois = []
        paths = []

        for i in range(self.num_simulations):
            accuracy, profit, roi, path = self.run_simulation(validation_data)
            accuracies.append(accuracy)
            profits.append(profit)
            rois.append(roi)
            if store_paths:
                paths.append(path)

        results = {
            "accuracy": {
                "mean": np.mean(accuracies),
                "std": np.std(accuracies),
                "ci_lower": np.percentile(accuracies, 2.5),
                "ci_upper": np.percentile(accuracies, 97.5),
            },
            "profit": {
                "mean": np.mean(profits),
                "std": np.std(profits),
                "ci_lower": np.percentile(profits, 2.5),
                "ci_upper": np.percentile(profits, 97.5),
            },
            "roi": {
                "mean": np.mean(rois),
                "std": np.std(rois),
                "ci_lower": np.percentile(rois, 2.5),
                "ci_upper": np.percentile(rois, 97.5),
            },
        }

        if store_paths:
            results["paths"] = paths

        return results

    def save_results(self, results: Dict, filename: str = None) -> None:
        """
        Save simulation results to file

        Args:
            results: Dictionary of simulation results
            filename: Optional custom filename, otherwise uses timestamp
        """
        if filename is None:
            filename = (
                f"results/monte_carlo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )

        with open(filename, "w") as f:
            f.write("Monte Carlo Simulation Results\n")
            f.write("============================\n\n")

            for metric in ["accuracy", "profit", "roi"]:
                f.write(f"{metric.title()} Statistics:\n")
                f.write(f"Mean: {results[metric]['mean']:.4f}\n")
                f.write(f"Std Dev: {results[metric]['std']:.4f}\n")
                f.write(
                    f"95% CI: [{results[metric]['ci_lower']:.4f}, {results[metric]['ci_upper']:.4f}]\n\n"
                )


def run_monte_carlo_analysis(num_simulations: int = 1000) -> Dict:
    """
    Helper function to run the Monte Carlo simulation

    Args:
        num_simulations: Number of simulations to run
    Returns:
        Dictionary of simulation results
    """
    validator = ModelValidator()
    validator.load_model()
    validator.load_validation_data()

    simulator = MonteCarloSimulator(validator, num_simulations)
    results = simulator.simulate()
    return results
