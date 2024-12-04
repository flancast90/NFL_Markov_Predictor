import copy
import numpy as np


class HMM:
    def __init__(
        self,
        transition_matrix,
        emission_matrix,
        observation_labels=None,
        matrices_are_log=False,
    ):
        if matrices_are_log:
            self.transition_matrix = transition_matrix
            self.emission_matrix = emission_matrix
        else:
            self.transition_matrix = np.log(transition_matrix + 1e-300)
            self.emission_matrix = np.log(emission_matrix + 1e-300)
        self.num_states = self.emission_matrix.shape[1]
        self.num_observations = self.emission_matrix.shape[0]
        self.observation_sequence = None
        self.alpha_matrix = []
        self.beta_matrix = []
        self.xi_matrix = []
        self.observation_labels = observation_labels
        self.observation_map = {}
        self.alpha_final = np.full(self.num_states, -np.inf)
        self.beta_final = np.full(self.num_states, -np.inf)
        self.state_probabilities = []

    def initialize_observation_labels(self):
        unique_observations = sorted(set(self.observation_sequence))
        self.observation_map = {obs: idx for idx, obs in enumerate(unique_observations)}
        return unique_observations

    def train(self, observation_sequence, num_iterations=10, verbose=True):
        self.observation_sequence = observation_sequence
        self.observation_labels = self.initialize_observation_labels()
        seq_len = len(self.observation_sequence)
        self.xi_matrix = np.full(
            (self.num_states, self.num_states, seq_len - 1), -np.inf
        )
        self.gamma_matrix = np.full((self.num_states, seq_len), -np.inf)
        for iter_num in range(num_iterations):
            if verbose:
                print(f"Iteration: {iter_num + 1}")
            self.expectation()
            self.maximization()

    def expectation(self):
        self.alpha_matrix = self.forward_recurse(len(self.observation_sequence))
        self.beta_matrix = self.backward_recurse(0)
        self.compute_gamma()
        self.compute_xi()

    def compute_gamma(self):
        for t in range(len(self.observation_sequence)):
            denominator = np.logaddexp.reduce(
                [
                    self.alpha_matrix[state, t] + self.beta_matrix[state, t]
                    for state in range(self.num_states)
                ]
            )
            for state in range(self.num_states):
                self.gamma_matrix[state, t] = (
                    self.alpha_matrix[state, t]
                    + self.beta_matrix[state, t]
                    - denominator
                )

    def compute_xi(self):
        for t in range(1, len(self.observation_sequence)):
            for curr_state in range(self.num_states):
                for prev_state in range(self.num_states):
                    self.xi_matrix[prev_state, curr_state, t - 1] = (
                        self.compute_xi_value(t, prev_state, curr_state)
                    )

    def compute_xi_value(self, time, prev_state, curr_state):
        alpha = self.alpha_matrix[prev_state, time - 1]
        transition = self.transition_matrix[curr_state + 1, prev_state + 1]
        beta = self.beta_matrix[curr_state, time]
        emission = self.emission_matrix[
            self.observation_map[self.observation_sequence[time]], curr_state
        ]
        denominator = np.logaddexp.reduce(
            [
                self.alpha_matrix[s, time - 1] + self.beta_matrix[s, time - 1]
                for s in range(self.num_states)
            ]
        )
        return alpha + transition + beta + emission - denominator

    def maximization(self):
        self.compute_state_probabilities()
        for state in range(self.num_states):
            self.transition_matrix[state + 1, 0] = self.gamma_matrix[state, 0]
            self.transition_matrix[-1, state + 1] = (
                self.gamma_matrix[state, -1] - self.state_probabilities[state]
            )
            for next_state in range(self.num_states):
                self.transition_matrix[next_state + 1, state + 1] = (
                    self.estimate_transition_probability(state, next_state)
                )
            for obs in range(self.num_observations):
                self.emission_matrix[obs, state] = self.estimate_emission_probability(
                    state, obs
                )

    def compute_state_probabilities(self):
        self.state_probabilities = np.logaddexp.reduce(self.gamma_matrix, axis=1)

    def estimate_transition_probability(self, from_state, to_state):
        return (
            np.logaddexp.reduce(self.xi_matrix[from_state, to_state])
            - self.state_probabilities[from_state]
        )

    def estimate_emission_probability(self, state, observation_idx):
        if observation_idx >= len(self.observation_labels):
            return -np.inf
        observation = self.observation_labels[observation_idx]
        matching_times = [
            t for t, obs in enumerate(self.observation_sequence) if obs == observation
        ]
        if not matching_times:
            return -np.inf
        return (
            np.logaddexp.reduce([self.gamma_matrix[state, t] for t in matching_times])
            - self.state_probabilities[state]
        )

    def backward_recurse(self, start_idx):
        seq_len = len(self.observation_sequence)
        beta = np.full((self.num_states, seq_len), -np.inf)
        for state in range(self.num_states):
            beta[state, seq_len - 1] = self.transition_matrix[
                self.num_states + 1, state + 1
            ]
        for t in range(seq_len - 2, -1, -1):
            for state in range(self.num_states):
                beta[state, t] = self.compute_beta(t, beta, state)
                if t == 0:
                    self.beta_final[state] = self.compute_beta(
                        t, beta, 0, is_final=True
                    )
        return beta

    def compute_beta(self, time_idx, beta_matrix, state, is_final=False):
        probabilities = []
        for next_state in range(self.num_states):
            observation = self.observation_sequence[time_idx + 1]
            transition = (
                self.transition_matrix[next_state + 1, 0]
                if is_final
                else self.transition_matrix[next_state + 1, state + 1]
            )
            emission = self.emission_matrix[
                self.observation_map[observation], next_state
            ]
            beta_val = beta_matrix[next_state, time_idx + 1]
            probabilities.append(transition + emission + beta_val)
        return np.logaddexp.reduce(probabilities)

    def forward_recurse(self, end_idx):
        seq_len = len(self.observation_sequence)
        alpha = np.full((self.num_states, seq_len), -np.inf)
        for state in range(self.num_states):
            observation = self.observation_sequence[0]
            initial_prob = self.transition_matrix[state + 1, 0]
            emission_prob = self.emission_matrix[
                self.observation_map[observation], state
            ]
            alpha[state, 0] = initial_prob + emission_prob
        for t in range(1, seq_len):
            for state in range(self.num_states):
                alpha[state, t] = self.compute_alpha(t, alpha, state)
        if end_idx == seq_len:
            for state in range(self.num_states):
                self.alpha_final[state] = self.compute_alpha(
                    end_idx, alpha, state, is_final=True
                )
        return alpha

    def compute_alpha(self, time_idx, alpha_matrix, curr_state, is_final=False):
        probabilities = []
        for prev_state in range(self.num_states):
            if not is_final:
                obs_idx = self.observation_map[self.observation_sequence[time_idx]]
                transition = self.transition_matrix[curr_state + 1, prev_state + 1]
                emission = self.emission_matrix[obs_idx, curr_state]
                probabilities.append(
                    alpha_matrix[prev_state, time_idx - 1] + transition + emission
                )
            else:
                transition = self.transition_matrix[self.num_states, prev_state + 1]
                probabilities.append(
                    alpha_matrix[prev_state, time_idx - 1] + transition
                )
        return np.logaddexp.reduce(probabilities)

    def likelihood(self, new_sequence):
        model_copy = copy.deepcopy(self)
        model_copy.observation_sequence = new_sequence
        new_labels = set(new_sequence) - set(model_copy.observation_labels)
        model_copy.observation_labels.extend(new_labels)
        model_copy.initialize_observation_labels()
        model_copy.forward_recurse(len(new_sequence))
        return np.logaddexp.reduce(model_copy.alpha_final)


if __name__ == "__main__":
    emission = np.array(
        [
            [0.319577735124672, 1.1820330969258756e-13],
            [0.247600767754272, 1.1820330969258756e-13],
            [0.432821497120768, 1.1820330969258756e-13],
            [9.596928982720001e-14, 0.43853427895961805],
            [9.596928982720001e-14, 0.24940898345147794],
            [9.596928982720001e-14, 0.31205673758854935],
        ]
    )

    transmission = np.array(
        [
            [0, 0, 0, 0],
            [0, 0.5422264875239842, 0.564497041420103, 0],
            [0, 0.4577735124760157, 0.4355029585798969, 0],
            [0, 0, 0, 0],
        ]
    )
    observations = [
        "close_win",
        "loss",
        "loss",
        "close_win",
        "loss",
        "close_win",
        "loss",
        "close_win",
        "close_win",
        "loss",
        "big_win",
        "loss",
        "loss",
        "big_win",
        "big_win",
        "big_win",
        "close_win",
        "big_win",
        "big_win",
        "big_win",
        "loss",
        "big_win",
        "close_win",
        "big_win",
        "big_win",
        "big_win",
        "close_win",
        "loss",
        "loss",
        "close_win",
        "loss",
        "close_win",
        "close_win",
    ]
    model = HMM(
        transmission,
        emission,
        ["big_win", "win", "close_win", "close_loss", "loss", "big_loss"],
    )
    model.train(observations)
    print(
        "Model transmission probabilities:\n{}".format(np.exp(model.transition_matrix))
    )
    print("Model emission probabilities:\n{}".format(np.exp(model.emission_matrix)))
    new_seq = ["big_win", "close_win", "loss"]
    print("Finding likelihood for {}".format(new_seq))
    likelihood = model.likelihood(new_seq)
    print("Log-likelihood: {}".format(likelihood))
