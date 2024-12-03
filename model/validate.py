import json
import numpy as np
from datetime import datetime
from markov import HMM


class ModelValidator:
    def __init__(self):
        self.hmm = None
        self.states = None
        self.observations = None
        self.validation_data = None

    def load_model(self, model_file: str = "model/saves/trained_model.json"):
        """Load trained model parameters"""
        with open(model_file, "r") as f:
            model_data = json.load(f)

        base_transition = np.array(model_data["transition_matrix"])
        emission_matrix = np.array(model_data["emission_matrix"])
        self.states = model_data["states"]
        self.observations = model_data["observations"]

        # Create padded transition matrix
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
        )

    def load_validation_data(self, validation_file: str = "data/validation_set.json"):
        """Load validation dataset"""
        with open(validation_file, "r") as f:
            validation_data = json.load(f)
            self.validation_data = validation_data["data"]

    def get_observation(self, home_score: int, away_score: int) -> str:
        """Convert score difference to observation category"""
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

    def validate(self):
        """Run validation and compute metrics"""
        if self.hmm is None or self.validation_data is None:
            raise ValueError("Must load model and validation data before validating")

        correct_predictions = 0
        total_predictions = len(self.validation_data)
        predictions = []
        actuals = []

        for game in self.validation_data:
            # Get actual observation
            observation = self.get_observation(game["home_score"], game["away_score"])
            actual_state = (
                "home_win" if game["home_score"] > game["away_score"] else "away_win"
            )

            # Make prediction using likelihood comparison
            home_likelihood = self.hmm.likelihood([observation])
            away_likelihood = self.hmm.likelihood([observation])
            predicted_state = (
                "home_win" if home_likelihood > away_likelihood else "away_win"
            )

            predictions.append(predicted_state)
            actuals.append(actual_state)

            if predicted_state == actual_state:
                correct_predictions += 1

        accuracy = correct_predictions / total_predictions

        # Calculate confusion matrix
        confusion_matrix = np.zeros((2, 2))
        for pred, actual in zip(predictions, actuals):
            pred_idx = self.states.index(pred)
            actual_idx = self.states.index(actual)
            confusion_matrix[actual_idx][pred_idx] += 1

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"results/results_{timestamp}.txt", "w") as f:
            f.write("Model Validation Results\n")
            f.write("=======================\n\n")
            f.write(f"Total predictions: {total_predictions}\n")
            f.write(f"Correct predictions: {correct_predictions}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n\n")

            f.write("Confusion Matrix\n")
            f.write("---------------\n")
            f.write("Predicted ->      Home Win    Away Win\n")
            f.write("Actual |\n")
            f.write(
                f"Home Win     {confusion_matrix[0][0]:10.0f} "
                f"{confusion_matrix[0][1]:10.0f}\n"
            )
            f.write(
                f"Away Win     {confusion_matrix[1][0]:10.0f} "
                f"{confusion_matrix[1][1]:10.0f}\n\n"
            )

            # Calculate additional metrics
            true_pos = confusion_matrix[0][0]  # True positives for home wins
            false_pos = confusion_matrix[1][0]  # False positives for home wins
            false_neg = confusion_matrix[0][1]  # False negatives for home wins

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

        return accuracy, confusion_matrix
