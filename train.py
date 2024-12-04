import json
import numpy as np
from markov import HMM
from .utils import get_historical_mov, get_game_observation


class ModelTrainer:
    def __init__(self):
        self.hmm = None
        self.states = ["home_win", "away_win"]
        self.observations = [
            "strong_history",
            "positive_history",
            "neutral_history",
            "negative_history",
            "weak_history",
        ]
        self.train_data = None

    def load_training_data(self, train_file: str = "data/train_set.json"):
        """
        Load training data and matrices from JSON file.

        Args:
        train_file: The location of the dataset used for training the model
        """
        with open(train_file, "r") as f:
            train_data = json.load(f)

        transition_matrix = np.array(train_data["transition_matrix"])
        emission_matrix = np.array(train_data["emission_matrix"])
        self.train_data = train_data["data"]

        # Initialize HMM with the matrices
        self.hmm = HMM(
            transition_matrix=transition_matrix,
            emission_matrix=emission_matrix,
            observation_labels=self.observations,
        )

    def train(self, num_iterations: int = 10):
        """
        Train the HMM model on the loaded data
        
        Args: 
            num_iterations: How many times the model is trained on the data
        """
        if self.hmm is None:
            raise ValueError("Must load training data before training")

        # Sort games by date for historical calculations
        sorted_games = sorted(self.train_data, key=lambda x: x["date"])

        # Extract observation sequence from training data
        observation_sequence = []
        for game in sorted_games:
            home_mov = get_historical_mov(game["home_team"], game["date"], sorted_games)
            away_mov = get_historical_mov(game["away_team"], game["date"], sorted_games)
            obs = get_game_observation(home_mov, away_mov)
            observation_sequence.append(obs)

        # Store original matrices
        original_transition = self.hmm.transition_matrix.copy()
        original_emission = self.hmm.emission_matrix.copy()

        # Train HMM using observation sequence
        self.hmm.train(observation_sequence, num_iterations=num_iterations)

        # Check if training produced valid matrices
        if np.all(self.hmm.transition_matrix == 0) or np.all(
            self.hmm.emission_matrix == 0
        ):
            print("Warning: Training produced invalid matrices, reverting to original")
            self.hmm.transition_matrix = original_transition
            self.hmm.emission_matrix = original_emission

    def save_model(self, output_file: str = "model/saves/trained_model.json"):
        """
        Save the trained model parameters.
        
        Args:
            output_file: The location the data should be output to.
        """
        if self.hmm is None:
            raise ValueError("No trained model to save")

        # If matrices are all zeros, something went wrong
        if np.all(self.hmm.transition_matrix == 0) or np.all(
            self.hmm.emission_matrix == 0
        ):
            print("Warning: Matrices contain all zeros, using original")
            # Load original matrices from training data
            with open("data/train_set.json", "r") as f:
                train_data = json.load(f)
                transition_matrix = np.array(train_data["transition_matrix"])
                emission_matrix = np.array(train_data["emission_matrix"])
        else:
            transition_matrix = self.hmm.transition_matrix
            emission_matrix = self.hmm.emission_matrix

        model_data = {
            "transition_matrix": transition_matrix.tolist(),
            "emission_matrix": emission_matrix.tolist(),
            "states": self.states,
            "observations": self.observations,
        }

        print("Saving model with:")
        print("Transition matrix shape:", transition_matrix.shape)
        print("Emission matrix shape:", emission_matrix.shape)
        print("Transition matrix:\n", transition_matrix)
        print("Emission matrix:\n", emission_matrix)

        with open(output_file, "w") as f:
            json.dump(model_data, f)
