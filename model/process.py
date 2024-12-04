import json
import numpy as np
from typing import List, Dict
import random
from datetime import datetime
from .utils import get_historical_mov, get_game_observation, get_game_result


class ModelProcessor:
    def __init__(self):
        self.states = ["home_win", "away_win"]
        self.observations = [
            "strong_history",
            "positive_history",
            "neutral_history",
            "negative_history",
            "weak_history",
        ]
        self.state_to_index = {state: i for i, state in enumerate(self.states)}
        self.obs_to_index = {obs: i for i, obs in enumerate(self.observations)}

        # Initialize matrices
        self.transition_matrix = np.zeros((len(self.states) + 2, len(self.states) + 2))
        self.emission_matrix = np.zeros((len(self.observations), len(self.states)))

        self.train_data = []
        self.validation_data = []

    def load_data(self, filepath: str) -> List[Dict]:
        with open(filepath, "r") as f:
            data = json.load(f)
            return [
                game
                for game in data
                if game.get("home_score") is not None
                and game.get("away_score") is not None
                and game.get("home_ml") is not None
                and game.get("away_ml") is not None
                and game.get("home_ml") != "NA"
                and game.get("away_ml") != "NA"
            ]

    def compute_matrices(self, games: List[Dict]):
        # Initialize matrices with small non-zero values
        self.transition_matrix = np.full(
            (len(self.states) + 2, len(self.states) + 2), 0.1667
        )
        self.emission_matrix = np.full(
            (len(self.observations), len(self.states)), 0.1667
        )

        # Sort games by date
        sorted_games = sorted(games, key=lambda x: x["date"])

        for i, current_game in enumerate(sorted_games):
            # Get actual game result
            current_state = get_game_result(
                current_game["home_score"], current_game["away_score"]
            )

            # Calculate historical MOVs
            home_mov = get_historical_mov(
                current_game["home_team"], current_game["date"], sorted_games
            )
            away_mov = get_historical_mov(
                current_game["away_team"], current_game["date"], sorted_games
            )

            # Get observation based on historical MOVs
            current_obs = get_game_observation(home_mov, away_mov)

            state_idx = self.state_to_index[current_state] + 1
            obs_idx = self.obs_to_index[current_obs]
            self.emission_matrix[obs_idx][state_idx - 1] += 1

            # Update transition matrix (same as before)
            if i == 0:
                self.transition_matrix[state_idx][0] += 1
            elif i == len(games) - 1:
                self.transition_matrix[-1][state_idx] += 1
            else:
                next_game = sorted_games[i + 1]
                next_state = get_game_result(
                    next_game["home_score"], next_game["away_score"]
                )
                self.transition_matrix[self.state_to_index[next_state] + 1][
                    state_idx
                ] += 1

        # Normalize matrices
        for j in range(len(self.states) + 2):
            col_sum = np.sum(self.transition_matrix[:, j])
            if col_sum > 0:
                self.transition_matrix[:, j] = self.transition_matrix[:, j] / col_sum

        for j in range(len(self.states)):
            col_sum = np.sum(self.emission_matrix[:, j])
            if col_sum > 0:
                self.emission_matrix[:, j] = self.emission_matrix[:, j] / col_sum

    def split_data(self, data: List[Dict], train_ratio: float = 0.7):
        random.shuffle(data)
        split_idx = int(len(data) * train_ratio)
        self.train_data = data[:split_idx]
        self.validation_data = data[split_idx:]

    def process_and_save(self, input_file: str = "data/nfl_dataset.json"):
        data = self.load_data(input_file)
        self.split_data(data)
        self.compute_matrices(self.train_data)

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
