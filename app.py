import streamlit as st
import json
from graphviz import Digraph
import numpy as np
import pandas as pd
import os

def load_hmm_model_from_file(file_path):
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            st.error(f"Could not find file: {file_path}")
            return None, None, None, None
            
        # Read and parse the JSON file
        with open(file_path, 'r') as file:
            model_data = json.load(file)
            
        # Convert matrices to numpy arrays for better handling
        transition_matrix = np.array(model_data["transition_matrix"])
        emission_matrix = np.array(model_data["emission_matrix"])
        states = model_data["states"]
        observations = model_data["observations"]
        
        return transition_matrix, emission_matrix, states, observations
        
    except json.JSONDecodeError as e:
        st.error(f"Error parsing JSON file: {str(e)}")
        return None, None, None, None
    except KeyError as e:
        st.error(f"Missing required key in JSON file: {str(e)}")
        return None, None, None, None
    except Exception as e:
        st.error(f"Unexpected error loading model: {str(e)}")
        return None, None, None, None

def create_hmm_visualization(transition_matrix, emission_matrix, states, observations):
    # Initialize a new directed graph with specific styling
    dot = Digraph(comment='Hidden Markov Model')
    dot.attr(rankdir='LR')
    
    dot.attr('node', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='10')
    
    # Create a subgraph for states to keep them aligned
    with dot.subgraph(name='cluster_states') as c:
        c.attr(rank='same')
        for state in states:
            c.node(f"state_{state}", 
                  state.replace('_', ' ').title(),
                  shape='circle',
                  style='filled',
                  fillcolor='lightblue',
                  width='1.5')
    
    # Create a subgraph for observations to keep them aligned
    with dot.subgraph(name='cluster_observations') as c:
        c.attr(rank='same')
        for obs in observations:
            c.node(f"obs_{obs}",
                  obs.replace('_', ' ').title(),
                  shape='box',
                  style='filled',
                  fillcolor='red',
                  width='1.5')
    
    # Add transition edges between states
    for i, from_state in enumerate(states):
        for j, to_state in enumerate(states):
            prob = transition_matrix[i][j]
            if prob > 0.001:  # Only show significant probabilities
                dot.edge(f"state_{from_state}",
                        f"state_{to_state}",
                        label=f"{prob:.2f}",
                        penwidth='2')
    
    # Add emission edges from states to observations
    for i, state in enumerate(states):
        for j, obs in enumerate(observations):
            prob = emission_matrix[j][i]
            if prob > 0.001:  # Only show significant probabilities
                dot.edge(f"state_{state}",
                        f"obs_{obs}",
                        label=f"{prob:.2f}",
                        style='dashed',
                        penwidth='1.5')
    
    return dot

def main():
    st.title("Hidden Markov Model Visualization")
    
    st.write("""
    This application creates a visual representation of your Hidden Markov Model from the trained_model.json file. 
    The visualization shows:
    - States (blue circles): The hidden states in your model
    - Observations (green rectangles): The possible observable outcomes
    - Solid arrows: Transition probabilities between states
    - Dashed arrows: Emission probabilities from states to observations
    """)
    
    # Load the model from the JSON file
    file_path = "model\\saves\\trained_model.json"
    transition_matrix, emission_matrix, states, observations = load_hmm_model_from_file(file_path)
    
    if all(v is not None for v in [transition_matrix, emission_matrix, states, observations]):
        # Create and display the visualization
        dot = create_hmm_visualization(transition_matrix, emission_matrix, states, observations)
        st.graphviz_chart(dot)
        
        # Show detailed model information in an expandable section
        with st.expander("View Model Details"):
            # Display transition probabilities
            st.subheader("Transition Probabilities")
            transition_df = pd.DataFrame(
                transition_matrix,
                columns=[s.replace('_', ' ').title() for s in states],
                index=[s.replace('_', ' ').title() for s in states]
            )
            st.dataframe(transition_df)
            
            # Display emission probabilities
            st.subheader("Emission Probabilities")
            emission_df = pd.DataFrame(
                emission_matrix,
                columns=[s.replace('_', ' ').title() for s in states],
                index=[o.replace('_', ' ').title() for o in observations]
            )
            st.dataframe(emission_df)
    else:
        st.warning("""
        Please ensure that:
        1. The file 'trained_model.json' is in the same directory as this script
        2. The file contains valid JSON data with the required fields
        3. You have permission to read the file
        """)

if __name__ == "__main__":
    main()