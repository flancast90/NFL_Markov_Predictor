import json
import numpy as np
from typing import List, Dict
import random


class ModelProcessor:
    def __init__(self):
        self.states = ["home_win", "away_win"]
        self.observations = [
            "big_win",
            "win",
            "close_win",
            "close_loss",
            "loss",
            "big_loss",
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
        else:
            return "away_win"

    def get_point_difference_observation(self, home_score: int, away_score: int) -> str:
        point_diff = home_score - away_score

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

        return obs

    def compute_matrices(self, games: List[Dict]):
        # Initialize matrices with small values
        self.transition_matrix = np.full((len(self.states), len(self.states)), 1e-10)
        self.emission_matrix = np.full(
            (len(self.observations), len(self.states)), 1e-10
        )

        # Count occurrences for transitions and emissions
        state_counts = {state: 0 for state in self.states}

        for i in range(len(games)):
            current_game = games[i]
            current_state = self.get_game_result(
                current_game["home_score"], current_game["away_score"]
            )
            current_obs = self.get_point_difference_observation(
                current_game["home_score"], current_game["away_score"]
            )

            # Update state counts
            state_counts[current_state] += 1

            # Update emission matrix
            state_idx = self.state_to_index[current_state]
            obs_idx = self.obs_to_index[current_obs]
            self.emission_matrix[obs_idx][state_idx] += 1

            # Update transition matrix if not last game
            if i < len(games) - 1:
                next_game = games[i + 1]
                next_state = self.get_game_result(
                    next_game["home_score"], next_game["away_score"]
                )
                self.transition_matrix[self.state_to_index[current_state]][
                    self.state_to_index[next_state]
                ] += 1

        # Normalize matrices
        # Normalize transition matrix (row-wise)
        for i in range(len(self.states)):
            row_sum = np.sum(self.transition_matrix[i])
            if row_sum > 0:
                self.transition_matrix[i] = self.transition_matrix[i] / row_sum

        # Normalize emission matrix (column-wise)
        for j in range(len(self.states)):
            state = self.states[j]
            if state_counts[state] > 0:
                self.emission_matrix[:, j] = (
                    self.emission_matrix[:, j] / state_counts[state]
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
