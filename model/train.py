import json
import numpy as np
from typing import Dict, List
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

        base_transition = np.array(train_data["transition_matrix"])
        base_emission = np.array(train_data["emission_matrix"])

        # Create padded transition matrix
        padded_transition = np.full(
            (len(base_transition) + 2, len(base_transition) + 2), 1e-10
        )

        # Copy the base transition probabilities to the middle of the padded matrix
        padded_transition[1:-1, 1:-1] = base_transition

        # Calculate initial probabilities (from start state to each state)
        # Use the first state distribution from base_transition
        padded_transition[1:-1, 0] = base_transition[0, :]

        # Calculate final probabilities (from each state to end state)
        # Use average of outgoing probabilities
        padded_transition[-1, 1:-1] = np.mean(base_transition, axis=1)

        # Normalize the start and end state probabilities
        padded_transition[1:-1, 0] /= np.sum(padded_transition[1:-1, 0])
        padded_transition[-1, 1:-1] /= np.sum(padded_transition[-1, 1:-1])

        self.states = train_data["states"]
        self.observations = train_data["observations"]
        self.train_data = train_data["data"]

        # Debug prints
        print("Base transition matrix shape:", base_transition.shape)
        print("Base emission matrix shape:", base_emission.shape)
        print("Padded transition matrix shape:", padded_transition.shape)
        print("Base transition matrix:\n", base_transition)
        print("Base emission matrix:\n", base_emission)
        print("Padded transition matrix:\n", padded_transition)

        # Initialize HMM with the matrices
        self.hmm = HMM(
            transition_matrix=padded_transition,
            emission_matrix=base_emission,
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
            print(
                "Warning: Training produced invalid matrices, reverting to original matrices"
            )
            self.hmm.transition_matrix = original_transition
            self.hmm.emission_matrix = original_emission

    def save_model(self, output_file: str = "model/saves/trained_model.json"):
        """Save the trained model parameters"""
        if self.hmm is None:
            raise ValueError("No trained model to save")

        # Extract the core transition matrix (remove padding)
        core_transition = self.hmm.transition_matrix[1:-1, 1:-1]

        # If matrices are all zeros, something went wrong
        if np.all(core_transition == 0) or np.all(self.hmm.emission_matrix == 0):
            print("Warning: Matrices contain all zeros, using original matrices")
            # Load original matrices from training data
            with open("data/train_set.json", "r") as f:
                train_data = json.load(f)
                core_transition = np.array(train_data["transition_matrix"])
                emission_matrix = np.array(train_data["emission_matrix"])
        else:
            emission_matrix = self.hmm.emission_matrix

        model_data = {
            "transition_matrix": core_transition.tolist(),
            "emission_matrix": emission_matrix.tolist(),
            "states": self.states,
            "observations": self.observations,
        }

        print("Saving model with:")
        print("Transition matrix shape:", core_transition.shape)
        print("Emission matrix shape:", emission_matrix.shape)
        print("Transition matrix:\n", core_transition)
        print("Emission matrix:\n", emission_matrix)

        with open(output_file, "w") as f:
            json.dump(model_data, f)
