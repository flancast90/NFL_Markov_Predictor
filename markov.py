import numpy as np


class HMM:

    def __init__(self, transition_matrix, emission_matrix, observation_labels=None):
        self.transition_matrix = transition_matrix
        self.emission_matrix = emission_matrix
        self.num_states = self.emission_matrix.shape[1]
        self.num_observations = self.emission_matrix.shape[0]
        self.observation_sequence = None
        self.alpha_matrix = []
        self.beta_matrix = []
        self.xi_matrix = []
        self.observation_labels = observation_labels
        self.observation_map = {}
        self.alpha_final = [0, 0]
        self.beta_final = [0, 0]
        self.state_probabilities = []

        if observation_labels is None and self.observation_sequence is not None:
            self.observation_labels = self.initialize_observation_labels()

    def initialize_observation_labels(self):
        unique_observations = list(set(list(self.observation_sequence)))
        unique_observations.sort()
        for idx, obs in enumerate(unique_observations):
            self.observation_map[obs] = idx
        return unique_observations

    def train(self, observation_sequence, num_iterations=10, verbose=True):
        self.observation_sequence = observation_sequence
        self.observation_labels = self.initialize_observation_labels()
        seq_len = len(self.observation_sequence)
        self.xi_matrix = [
            [[0.0] * (seq_len - 1) for _ in range(self.num_states)]
            for _ in range(self.num_states)
        ]
        self.gamma_matrix = [[0.0] * seq_len for _ in range(self.num_states)]

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
        self.gamma_matrix = [[0, 0] for _ in range(len(self.observation_sequence))]
        for t in range(len(self.observation_sequence)):
            denominator = sum(
                self.alpha_matrix[state][t] * self.beta_matrix[state][t]
                for state in range(2)
            )
            for state in range(2):
                numerator = self.alpha_matrix[state][t] * self.beta_matrix[state][t]
                # Add small epsilon to avoid division by zero
                self.gamma_matrix[t][state] = numerator / (denominator + 1e-10)

    def compute_xi(self):
        for t in range(1, len(self.observation_sequence)):
            for curr_state in range(self.num_states):
                for prev_state in range(self.num_states):
                    self.xi_matrix[prev_state][curr_state][t - 1] = (
                        self.compute_xi_value(t, prev_state, curr_state)
                    )

    def compute_xi_value(self, time, prev_state, curr_state):
        alpha = self.alpha_matrix[prev_state][time - 1]
        transition = self.transition_matrix[curr_state + 1][prev_state + 1]
        beta = self.beta_matrix[curr_state][time]
        emission = self.emission_matrix[
            self.observation_map[self.observation_sequence[time]]
        ][curr_state]
        denominator = sum(
            self.alpha_matrix[s][time - 1] * self.beta_matrix[s][time - 1]
            for s in range(2)
        )
        # Add small epsilon to avoid division by zero
        return (alpha * transition * beta * emission) / (denominator + 1e-10)

    def maximization(self):
        self.compute_state_probabilities()
        for state in range(self.num_states):
            self.transition_matrix[state + 1][0] = self.gamma_matrix[0][state]
            # Add small epsilon to avoid division by zero
            self.transition_matrix[-1][state + 1] = self.gamma_matrix[-1][state] / (
                self.state_probabilities[state] + 1e-10
            )

            for next_state in range(self.num_states):
                self.transition_matrix[next_state + 1][state + 1] = (
                    self.estimate_transition_probability(state, next_state)
                )

            for obs in range(self.num_observations):
                self.emission_matrix[obs][state] = self.estimate_emission_probability(
                    state, obs
                )

    def compute_state_probabilities(self):
        self.state_probabilities = [
            sum(row[state] for row in self.gamma_matrix)
            for state in range(self.num_states)
        ]

    def estimate_transition_probability(self, from_state, to_state):
        # Add small epsilon to avoid division by zero
        return sum(self.xi_matrix[from_state][to_state]) / (
            self.state_probabilities[from_state] + 1e-10
        )

    def estimate_emission_probability(self, state, observation_idx):
        observation = self.observation_labels[observation_idx]
        matching_times = [
            t for t, obs in enumerate(self.observation_sequence) if obs == observation
        ]
        # Add small epsilon to avoid division by zero
        return sum(self.gamma_matrix[t][state] for t in matching_times) / (
            self.state_probabilities[state] + 1e-10
        )

    def backward_recurse(self, start_idx):
        seq_len = len(self.observation_sequence)
        beta = [[0.0] * seq_len for _ in range(self.num_states)]

        # Initialize final values
        for state in range(self.num_states):
            beta[state][seq_len - 1] = self.transition_matrix[self.num_states + 1][
                state + 1
            ]

        # Iterate backwards
        for t in range(seq_len - 2, -1, -1):
            for state in range(self.num_states):
                beta[state][t] = self.compute_beta(t, beta, state)
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
                self.transition_matrix[next_state + 1][0]
                if is_final
                else self.transition_matrix[next_state + 1][state + 1]
            )
            emission = self.emission_matrix[self.observation_map[observation]][
                next_state
            ]
            beta_val = beta_matrix[next_state][time_idx + 1]
            probabilities.append(transition * emission * beta_val)
        return sum(probabilities)

    def forward_recurse(self, end_idx):
        seq_len = len(self.observation_sequence)
        alpha = [[0.0] * seq_len for _ in range(self.num_states)]

        # Initialize first values
        for state in range(self.num_states):
            observation = self.observation_sequence[0]
            initial_prob = self.transition_matrix[state + 1][0]
            emission_prob = self.emission_matrix[self.observation_map[observation]][
                state
            ]
            alpha[state][0] = initial_prob * emission_prob

        # Iterate forward
        for t in range(1, seq_len):
            for state in range(self.num_states):
                alpha[state][t] = self.compute_alpha(t, alpha, state)

        # Compute final values if needed
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
                transition = self.transition_matrix[curr_state + 1][prev_state + 1]
                emission = self.emission_matrix[obs_idx][curr_state]
                probabilities.append(
                    alpha_matrix[prev_state][time_idx - 1] * transition * emission
                )
            else:
                transition = self.transition_matrix[self.num_states][prev_state + 1]
                probabilities.append(
                    alpha_matrix[prev_state][time_idx - 1] * transition
                )
        return sum(probabilities)

    def likelihood(self, new_sequence):
        model_copy = HMM(self.transition_matrix, self.emission_matrix)
        model_copy.observation_sequence = new_sequence
        model_copy.observation_labels = model_copy.initialize_observation_labels()
        model_copy.forward_recurse(len(new_sequence))
        return sum(model_copy.alpha_final)
