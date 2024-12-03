import json
import numpy as np
from typing import List, Dict
import random


class ModelProcessor:
    def __init__(self):
        self.states = ["home_win", "away_win", "draw"]
        self.observations = [
            "home+large",
            "home+small",
            "draw",
            "away+small",
            "away+large",
        ]
        self.state_to_index = {state: i for i, state in enumerate(self.states)}
        self.obs_to_index = {obs: i for i, obs in enumerate(self.observations)}

        # Initialize matrices
        self.transition_matrix = np.zeros((len(self.states), len(self.states)))
        self.emission_matrix = np.zeros((len(self.observations), len(self.states)))

        self.train_data = []
        self.validation_data = []

    def load_data(self, filepath: str) -> List[Dict]:
        with open(filepath, "r") as f:
            data = json.load(f)
            # Filter out games with missing/None scores
            return [
                game
                for game in data
                if game.get("home_score") is not None
                and game.get("away_score") is not None
            ]

    def get_game_result(self, home_score: int, away_score: int) -> str:
        if home_score > away_score:
            return "home_win"
        elif away_score > home_score:
            return "away_win"
        return "draw"

    def get_point_difference_observation(self, home_score: int, away_score: int) -> str:
        diff = home_score - away_score
        if diff >= 10:
            return "home+large"
        elif diff > 0:
            return "home+small"
        elif diff == 0:
            return "draw"
        elif diff > -10:
            return "away+small"
        else:
            return "away+large"

    def compute_matrices(self, games: List[Dict]):
        # Reset matrices
        self.transition_matrix.fill(0)
        self.emission_matrix.fill(0)

        # Process games to get sequences
        game_states = []
        game_observations = []

        for game in games:
            state = self.get_game_result(game["home_score"], game["away_score"])
            observation = self.get_point_difference_observation(
                game["home_score"], game["away_score"]
            )

            game_states.append(state)
            game_observations.append(observation)

        # Compute transition matrix
        for i in range(len(game_states) - 1):
            curr_state = game_states[i]
            next_state = game_states[i + 1]
            self.transition_matrix[
                self.state_to_index[curr_state], self.state_to_index[next_state]
            ] += 1

        # Normalize transition matrix
        row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
        self.transition_matrix = np.divide(
            self.transition_matrix,
            row_sums,
            out=np.zeros_like(self.transition_matrix),
            where=row_sums != 0,
        )

        # Compute emission matrix
        for state, obs in zip(game_states, game_observations):
            self.emission_matrix[
                self.obs_to_index[obs], self.state_to_index[state]
            ] += 1

        # Normalize emission matrix
        col_sums = self.emission_matrix.sum(axis=0, keepdims=True)
        self.emission_matrix = np.divide(
            self.emission_matrix,
            col_sums,
            out=np.zeros_like(self.emission_matrix),
            where=col_sums != 0,
        )

    def split_data(self, data: List[Dict], train_ratio: float = 0.7):
        random.shuffle(data)
        split_idx = int(len(data) * train_ratio)
        self.train_data = data[:split_idx]
        self.validation_data = data[split_idx:]

    def process_and_save(self, input_file: str = "data/nfl_dataset.json"):
        # Load data
        data = self.load_data(input_file)

        # Split into train/validation
        self.split_data(data)

        # Compute matrices using training data
        self.compute_matrices(self.train_data)

        # Save processed datasets
        with open("data/train_set.json", "w") as f:
            json.dump(
                {
                    "data": self.train_data,
                    "transition_matrix": self.transition_matrix.tolist(),
                    "emission_matrix": self.emission_matrix.tolist(),
                    "states": self.states,
                    "observations": self.observations,
                },
                f,
            )

        with open("data/validation_set.json", "w") as f:
            json.dump(
                {
                    "data": self.validation_data,
                    "states": self.states,
                    "observations": self.observations,
                },
                f,
            )
