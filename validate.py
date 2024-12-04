import json
import numpy as np
from datetime import datetime
from markov import HMM
from .utils import get_historical_mov, get_game_observation, get_game_result

strategy: str = "tail"  # Can be either "tail" or "fade"


class ModelValidator:
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
        self.validation_data = None

    def __american_to_decimal(self, american_odds: int) -> float:
        american_odds = int(american_odds)
        return (
            american_odds / 100 + 1 if american_odds > 0 else 1 - (100 / american_odds)
        )

    def load_model(self, model_file: str = "model/saves/trained_model.json"):
        """
        Loads the model from the given json file
        
        Args:
            model_file: The location of the file being loaded
        """
        with open(model_file, "r") as f:
            model_data = json.load(f)

        base_transition = np.array(model_data["transition_matrix"])
        emission_matrix = np.array(model_data["emission_matrix"])

        padded_transition = np.full(
            (len(base_transition) + 2, len(base_transition) + 2), 1e-10
        )
        padded_transition[1:-1, 1:-1] = base_transition
        padded_transition[1:-1, 0] = base_transition[0, :]
        padded_transition[-1, 1:-1] = np.mean(base_transition, axis=1)
        padded_transition[1:-1, 0] /= np.sum(padded_transition[1:-1, 0])
        padded_transition[-1, 1:-1] /= np.sum(padded_transition[-1, 1:-1])

        self.hmm = HMM(
            transition_matrix=padded_transition,
            emission_matrix=emission_matrix,
            observation_labels=self.observations,
            matrices_are_log=True,
        )

    def load_validation_data(self, validation_file: str = "data/validation_set.json"):
        with open(validation_file, "r") as f:
            validation_data = json.load(f)
            self.validation_data = validation_data["data"]

    def predict(self, home_team: str, away_team: str) -> str:
        """
        Predict winner based on team names.
        
        Args: 
            home_team: The name of the home team
            away_team: The name of the away team

        Returns:
            model_prediction: The expected result (home or away win)
        """
        if self.hmm is None:
            raise ValueError("Must load model before predicting")

        if self.validation_data is None:
            raise ValueError("Must load validation data before predicting")

        # Sort games by date for historical calculations
        sorted_games = sorted(self.validation_data, key=lambda x: x["date"])
        latest_date = sorted_games[-1]["date"]

        # Get historical MOVs for prediction
        home_mov = get_historical_mov(home_team, latest_date, sorted_games)
        away_mov = get_historical_mov(away_team, latest_date, sorted_games)
        observation = get_game_observation(home_mov, away_mov)

        obs_idx = self.observations.index(observation)
        home_state_idx = self.states.index("home_win")
        away_state_idx = self.states.index("away_win")

        home_likelihood = self.hmm.emission_matrix[obs_idx][home_state_idx]
        away_likelihood = self.hmm.emission_matrix[obs_idx][away_state_idx]

        model_prediction = (
            "home_win" if home_likelihood > away_likelihood else "away_win"
        )
        return (
            model_prediction
            if strategy == "tail"
            else ("away_win" if model_prediction == "home_win" else "home_win")
        )

    def validate(self):
        """
        Validate training data.
        
        Returns:
            accuracy: The accuracy of the model to the verification data.
            confusion_matrix: The confusion matrix obtained from the model and verification data.
            total_profit: The theoretical net profit resulting from using this model.
            roi: The return on investment this model gives.
        """
        if self.hmm is None or self.validation_data is None:
            raise ValueError("Must load model and validation data before validating")

        # Sort games by date for historical calculations
        sorted_games = sorted(self.validation_data, key=lambda x: x["date"])

        correct_predictions = 0
        total_predictions = len(sorted_games)
        predictions = []
        actuals = []
        total_profit = 0
        bets = 0

        for game in sorted_games:
            # Get historical MOVs for prediction
            home_mov = get_historical_mov(game["home_team"], game["date"], sorted_games)
            away_mov = get_historical_mov(game["away_team"], game["date"], sorted_games)
            observation = get_game_observation(home_mov, away_mov)

            actual_state = get_game_result(game["home_score"], game["away_score"])
            predicted_state = self.predict(game["home_team"], game["away_team"])

            predictions.append(predicted_state)
            actuals.append(actual_state)

            if predicted_state == "home_win":
                odds = self.__american_to_decimal(game["home_ml"])
                total_profit += odds - 1 if actual_state == "home_win" else -1
                bets += 1
            else:
                odds = self.__american_to_decimal(game["away_ml"])
                total_profit += odds - 1 if actual_state == "away_win" else -1
                bets += 1

            if predicted_state == actual_state:
                correct_predictions += 1

        accuracy = correct_predictions / total_predictions
        roi = (total_profit / bets) * 100 if bets > 0 else 0

        confusion_matrix = np.zeros((2, 2))
        for pred, actual in zip(predictions, actuals):
            pred_idx = self.states.index(pred)
            actual_idx = self.states.index(actual)
            confusion_matrix[actual_idx][pred_idx] += 1

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"results/results_{timestamp}.txt", "w") as f:
            f.write(f"Model Validation Results -> {strategy}ing strategy\n")
            f.write("=======================\n\n")
            f.write(f"Total predictions: {total_predictions}\n")
            f.write(f"Correct predictions: {correct_predictions}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Total profit: {total_profit:.2f} units\n")
            f.write(f"ROI: {roi:.2f}%\n\n")

            f.write("Confusion Matrix\n")
            f.write("---------------\n")
            f.write("Predicted ->      Home Win    Away Win\n")
            f.write("Actual |\n")
            f.write(
                f"Home Win     {confusion_matrix[0][0]:10.0f} {confusion_matrix[0][1]:10.0f}\n"
            )
            f.write(
                f"Away Win     {confusion_matrix[1][0]:10.0f} {confusion_matrix[1][1]:10.0f}\n\n"
            )

            true_pos = confusion_matrix[0][0]
            false_pos = confusion_matrix[1][0]
            false_neg = confusion_matrix[0][1]

            precision = (
                true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            )
            recall = (
                true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            )
            f1_score = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            f.write("Additional Metrics (for Home Win prediction)\n")
            f.write("----------------------------------------\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1_score:.4f}\n")

        return accuracy, confusion_matrix, total_profit, roi
