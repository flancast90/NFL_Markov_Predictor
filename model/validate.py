import json
import numpy as np
from datetime import datetime
from markov import HMM

strategy: str = "tail"  # Can be either "tail" or "fade"


class ModelValidator:
    def __init__(self):
        self.hmm = None
        self.states = None
        self.observations = None
        self.validation_data = None

    def __american_to_decimal(self, american_odds: int) -> float:
        american_odds = int(american_odds)
        return (
            american_odds / 100 + 1 if american_odds > 0 else 1 - (100 / american_odds)
        )

    def load_model(self, model_file: str = "model/saves/trained_model.json"):
        with open(model_file, "r") as f:
            model_data = json.load(f)

        base_transition = np.array(model_data["transition_matrix"])
        emission_matrix = np.array(model_data["emission_matrix"])
        self.states = model_data["states"]
        self.observations = model_data["observations"]

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

    def get_observation(self, home_score: int, away_score: int) -> str:
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

    def predict(self, observation: str) -> str:
        """Predict winner based on observation."""
        if self.hmm is None:
            raise ValueError("Must load model before predicting")

        obs_idx = self.observations.index(observation)
        home_state_idx = self.states.index("home_win")
        away_state_idx = self.states.index("away_win")

        # The emission matrix is shaped (n_observations, n_states)
        home_likelihood = self.hmm.emission_matrix[obs_idx][home_state_idx]
        away_likelihood = self.hmm.emission_matrix[obs_idx][away_state_idx]
        print(home_likelihood, away_likelihood)

        model_prediction = (
            "home_win" if home_likelihood > away_likelihood else "away_win"
        )
        return (
            model_prediction
            if strategy == "tail"
            else ("away_win" if model_prediction == "home_win" else "home_win")
        )

    def validate(self):
        if self.hmm is None or self.validation_data is None:
            raise ValueError("Must load model and validation data before validating")

        correct_predictions = 0
        total_predictions = len(self.validation_data)
        predictions = []
        actuals = []
        total_profit = 0
        bets = 0

        for game in self.validation_data:
            observation = self.get_observation(game["home_score"], game["away_score"])
            actual_state = (
                "home_win" if game["home_score"] > game["away_score"] else "away_win"
            )

            predicted_state = self.predict(observation)
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