import random
from markov import HMM
from collections import defaultdict

def generate_sequence(hmm_model, seq_len):
   
    sequence = []
    current_state = random.choices(range(hmm_model.num_states), weights=[hmm_model.transition_matrix[state + 1][0] for state in range(hmm_model.num_states)])[0]

    for _ in range(seq_len):
        observation = random.choices(hmm_model.observation_labels, weights=hmm_model.emission_matrix[:, current_state])[0]
        sequence.append(observation)
        next_state_probs = [hmm_model.transition_matrix[next_state + 1][current_state + 1] for next_state in range(hmm_model.num_states)]
        current_state = random.choices(range(hmm_model.num_states), weights=next_state_probs)[0]
    return sequence

def monte_carlo_simulation(hmm_model, num_sims, seq_len):
    sequences = []
    for _ in range(num_sims):
        seq, _ = generate_sequence(hmm_model, seq_len)
        sequences.append(seq)
    return sequences

def monte_carlo_graph(sequences):

    transition_counts = defaultdict(int)
    total_transitions = defaultdict(int)

    for sequence in sequences:
        for i in range(len(sequence) - 1):
            transition = (sequence[i], sequence[i + 1])
            transition_counts[transition] += 1
            total_transitions[sequence[i]] += 1

    adjacency_list = {}
    for (start, end), count in transition_counts.items():
        adjacency_list[(start, end)] = count / total_transitions[start]

    return adjacency_list