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

    def get_game_result(self, home_score: int, away_score: int) -> str:
        return "home_win" if home_score > away_score else "away_win"

    def get_point_difference_observation(self, home_score: int, away_score: int) -> str:
        point_diff = home_score - away_score
        if point_diff > 14:
            return "big_win"
        elif point_diff > 7:
            return "win"
        elif point_diff > 0:
            return "close_win"
        elif point_diff > -7:
            return "close_loss"
        elif point_diff > -14:
            return "loss"
        else:
            return "big_loss"

    def compute_matrices(self, games: List[Dict]):
        self.transition_matrix = np.full(
            (len(self.states) + 2, len(self.states) + 2), 0.1667
        )
        self.emission_matrix = np.full(
            (len(self.observations), len(self.states)), 0.1667
        )

        state_counts = {state: 0 for state in self.states}

        for i in range(len(games)):
            current_game = games[i]
            current_state = self.get_game_result(
                current_game["home_score"], current_game["away_score"]
            )
            current_obs = self.get_point_difference_observation(
                current_game["home_score"], current_game["away_score"]
            )

            state_counts[current_state] += 1

            state_idx = self.state_to_index[current_state] + 1  # +1 for start state
            obs_idx = self.obs_to_index[current_obs]
            self.emission_matrix[obs_idx][state_idx - 1] += 1

            if i == 0:
                self.transition_matrix[state_idx][0] += 1
            elif i == len(games) - 1:
                self.transition_matrix[-1][state_idx] += 1
            else:
                next_game = games[i + 1]
                next_state = self.get_game_result(
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
