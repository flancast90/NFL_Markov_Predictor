import json
import numpy as np
from markov import HMM


class ModelTrainer:
    def __init__(self):
        self.hmm = None
        self.states = None
        self.observations = None
        self.train_data = None

    def load_training_data(self, train_file: str = "data/train_set.json"):
        """Load training data and matrices from JSON file"""
        with open(train_file, "r") as f:
            train_data = json.load(f)

        transition_matrix = np.array(train_data["transition_matrix"])
        emission_matrix = np.array(train_data["emission_matrix"])

        self.states = train_data["states"]
        self.observations = train_data["observations"]
        self.train_data = train_data["data"]

        # Debug prints
        print("Transition matrix shape:", transition_matrix.shape)
        print("Emission matrix shape:", emission_matrix.shape)
        print("Transition matrix:\n", transition_matrix)
        print("Emission matrix:\n", emission_matrix)

        # Initialize HMM with the matrices
        self.hmm = HMM(
            transition_matrix=transition_matrix,
            emission_matrix=emission_matrix,
            observation_labels=self.observations,
        )

    def train(self, num_iterations: int = 10):
        """Train the HMM model on the loaded data"""
        if self.hmm is None:
            raise ValueError("Must load training data before training")

        # Extract observation sequence from training data
        observation_sequence = []
        for game in self.train_data:
            point_diff = game["home_score"] - game["away_score"]
            # Convert point difference to observation category
            if point_diff > 14:
                obs = "big_win"
            elif point_diff > 7:
                obs = "win"
            elif point_diff > 0:
                obs = "close_win"
            elif point_diff > -7:
                obs = "close_loss"
            elif point_diff > -14:
                obs = "loss"
            else:
                obs = "big_loss"
            observation_sequence.append(obs)

        # Store original matrices
        original_transition = self.hmm.transition_matrix.copy()
        original_emission = self.hmm.emission_matrix.copy()

        # Train HMM using observation sequence
        self.hmm.train(observation_sequence, num_iterations=num_iterations)

        # Check if training produced valid matrices, if not, revert to original
        if np.all(self.hmm.transition_matrix == 0) or np.all(
            self.hmm.emission_matrix == 0
        ):
            print("Warning: Training produced invalid matrices, reverting to original")
            self.hmm.transition_matrix = original_transition
            self.hmm.emission_matrix = original_emission

    def save_model(self, output_file: str = "model/saves/trained_model.json"):
        """Save the trained model parameters"""
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
