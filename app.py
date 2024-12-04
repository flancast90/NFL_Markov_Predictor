import streamlit as st
import json
import pickle
from graphviz import Digraph
import numpy as np
import pandas as pd
import os
import plotly.express as px
from datetime import datetime
from typing import List, Dict, Tuple

st.set_page_config(page_title="HMM Sports Predictor", layout="wide")

# Reads and parses the HMM model from a JSON file, converting log probabilities to regular probabilities for visualization.
def load_hmm_model_from_file(file_path):
    try:
        if not os.path.exists(file_path):
            st.error(f"Could not find file: {file_path}")
            return None, None, None, None
            
        with open(file_path, 'r') as file:
            model_data = json.load(file)
            
        # Get the 2x2 transition matrix and convert from log probabilities
        transition_matrix = np.array(model_data["transition_matrix"])[:2, :2]
        transition_matrix = np.exp(transition_matrix)
        
        # Convert emission matrix from log probabilities
        emission_matrix = np.exp(np.array(model_data["emission_matrix"]))
        
        states = model_data["states"]
        observations = model_data["observations"]
        
        return transition_matrix, emission_matrix, states, observations
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None
    
# Creates a visualization showing both state transitions and emission probabilities.
def create_hmm_visualization(transition_matrix, emission_matrix, states, observations):
    dot = Digraph(comment='Hidden Markov Model')
    dot.attr(rankdir='LR')
    
    with dot.subgraph(name='cluster_states') as c:
        c.attr(rank='same')
        for state in states:
            c.node(f"state_{state}", 
                  state.replace('_', ' ').title(),
                  shape='circle',
                  style='filled',
                  fillcolor='lightblue',
                  width='1.5')
    
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
            if prob > 0.01:
                prob_label = f"{prob*100:.1f}%"
                penwidth = str(1 + 2 * prob)
                dot.edge(f"state_{from_state}",
                        f"state_{to_state}",
                        label=prob_label,
                        penwidth=penwidth)
    
    # Add emission edges from states to observations
    for i, state in enumerate(states):
        for j, obs in enumerate(observations):
            prob = emission_matrix[j][i]
            if prob > 0.01:
                prob_label = f"{prob*100:.1f}%"
                dot.edge(f"state_{state}",
                        f"obs_{obs}",
                        label=prob_label,
                        style='dashed',
                        penwidth='1.0')
    
    return dot

# Creates a heatmap visualization of probability matrices using plotly.
def create_heatmap(matrix, labels_x, labels_y):
    fig = px.imshow(matrix,
                    labels=dict(x="To State", y="From State", color="Probability"),
                    x=labels_x,
                    y=labels_y,
                    color_continuous_scale="Blues")
    return fig

# Loads NFL game data from a JSON file and returns it as a list of game dictionaries.
def load_nfl_games(file_path: str) -> List[Dict]:
    try:
        with open(file_path, 'r') as file:
            games = json.load(file)
        return games
    except Exception as e:
        st.error(f"Error loading NFL games data: {str(e)}")
        return []
    
#Extracts a sorted list of unique team names from the games data.
def get_unique_teams(games: List[Dict]) -> List[str]:
    # Create a set of all teams (both home and away) and convert to sorted list
    teams = set()
    for game in games:
        teams.add(game['home_team'])
        teams.add(game['away_team'])
    return sorted(list(teams))

# Predicts the winner based on transition probabilities.
def predict_winner(home_team, away_team, transition_matrix, states):
    home_win_prob = transition_matrix[0][0]
    away_win_prob = transition_matrix[1][1]
    
    probabilities = {
        home_team: home_win_prob * 100,
        away_team: away_win_prob * 100
    }
    
    winner = home_team if home_win_prob > away_win_prob else away_team
    return winner, probabilities

# Calculates a team's win-loss record from the games data.
def get_team_history(games: List[Dict], team: str) -> Tuple[int, int]:
    wins = 0
    losses = 0
    
    for game in games:
        # Check if we have valid scores for this game
        try:
            home_score = game.get('home_score')
            away_score = game.get('away_score')
            
            # Skip games with missing scores
            if home_score is None or away_score is None:
                continue
                
            if game['home_team'] == team:
                if int(home_score) > int(away_score):
                    wins += 1
                elif int(home_score) < int(away_score):
                    losses += 1
                # Ties are not counted
            elif game['away_team'] == team:
                if int(away_score) > int(home_score):
                    wins += 1
                elif int(away_score) < int(home_score):
                    losses += 1
                
        except (TypeError, ValueError) as e:
            continue
                
    return wins, losses

# Creates and displays a table of recent matchups between two teams.
def display_matchup_history(games: List[Dict], home_team: str, away_team: str):
    matchups = [game for game in games if 
               (game['home_team'] == home_team and game['away_team'] == away_team) or
               (game['home_team'] == away_team and game['away_team'] == home_team)]
    
    if matchups:
        matchup_data = []
        for game in matchups[-5:]:
            # Handle possible missing scores
            try:
                home_score = game.get('home_score', 'N/A')
                away_score = game.get('away_score', 'N/A')
                
                if home_score != 'N/A' and away_score != 'N/A':
                    winner = game['home_team'] if int(home_score) > int(away_score) else game['away_team']
                else:
                    winner = 'Unknown'
                    
                matchup_data.append({
                    'Date': str(game['date']),
                    'Home': f"{game['home_team']} ({home_score})",
                    'Away': f"{game['away_team']} ({away_score})",
                    'Winner': winner
                })
            except (TypeError, ValueError):
                continue
        
        if matchup_data:
            st.dataframe(pd.DataFrame(matchup_data))
        else:
            st.info("No valid matchup data found between these teams")
    else:
        st.info("No recent matchups found between these teams")

def main():
    # Create sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", 
                           ["Model Visualization",
                            "Make Predictions",
                            "Model Analysis"])

    file_path = "model\\saves\\trained_model.json"
    transition_matrix, emission_matrix, states, observations = load_hmm_model_from_file(file_path)

    if all(v is not None for v in [transition_matrix, emission_matrix, states, observations]):
        if page == "Model Visualization":
            st.title("Hidden Markov Model Visualization")
            
            st.write("""
            This visualization shows the structure of your Hidden Markov Model:
            - States (blue circles): Hidden states representing game outcomes
            - Observations (red boxes): Observable game results
            - Solid arrows: Transition probabilities between states
            - Dashed arrows: Emission probabilities from states to observations
            """)
            
            # Display the HMM visualization
            dot = create_hmm_visualization(transition_matrix, emission_matrix, states, observations)
            st.graphviz_chart(dot)
            
            # Show probability matrices
            col1, col2 = st.columns(2)
            with col1:
                with st.expander("View Transition Probabilities"):
                    st.subheader("State Transitions (%)")
                    transition_df = pd.DataFrame(
                        transition_matrix * 100,
                        columns=[s.replace('_', ' ').title() for s in states],
                        index=[s.replace('_', ' ').title() for s in states]
                    )
                    st.dataframe(transition_df.round(2))
            
            with col2:
                with st.expander("View Emission Probabilities"):
                    st.subheader("Emissions (%)")
                    emission_df = pd.DataFrame(
                        emission_matrix * 100,
                        columns=[s.replace('_', ' ').title() for s in states],
                        index=[o.replace('_', ' ').title() for o in observations]
                    )
                    st.dataframe(emission_df.round(2))

        elif page == "Make Predictions":
            st.title("Predict Game Outcomes")
            
            # Load NFL games data
            games_file_path = "data\\nfl_dataset.json"
            games = load_nfl_games(games_file_path)
            
            if games:
                teams = get_unique_teams(games)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Home Team")
                    home_team = st.selectbox(
                        "Select Home Team",
                        teams,
                        key='home_team'
                    )
                    
                    if home_team:
                        wins, losses = get_team_history(games, home_team)
                        win_pct = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
                        st.write(f"Season Record: {wins}-{losses} ({win_pct:.1f}%)")
                
                with col2:
                    st.subheader("Away Team")
                    away_teams = [team for team in teams if team != home_team]
                    away_team = st.selectbox(
                        "Select Away Team",
                        away_teams,
                        key='away_team'
                    )
                    
                    if away_team:
                        wins, losses = get_team_history(games, away_team)
                        win_pct = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
                        st.write(f"Season Record: {wins}-{losses} ({win_pct:.1f}%)")
                
                # Only display prediction if both teams are selected
                if home_team and away_team:
                    # Get prediction using the transition matrix
                    predicted_winner, probabilities = predict_winner(home_team, away_team, transition_matrix, states)
                    win_probability = max(probabilities.values())
                    
                    # Display the prediction with team name and confidence
                    st.markdown("---")
                    winner_color = "green" if predicted_winner == home_team else "red"
                    st.markdown(f"### The winner will be: <span style='color:{winner_color}'>{predicted_winner}</span> ({win_probability:.1f}% confidence)", unsafe_allow_html=True)
                    st.markdown("---")
                    
                    # Show matchup history below the prediction
                    st.subheader("Recent Matchups")
                    display_matchup_history(games, home_team, away_team)


        elif page == "Model Analysis":
            st.title("Model Analysis")
            
            # Display heatmaps
            st.subheader("Transition Probabilities Heatmap")
            fig = create_heatmap(transition_matrix,
                               [s.replace('_', ' ').title() for s in states],
                               [s.replace('_', ' ').title() for s in states])
            st.plotly_chart(fig)
            
            st.subheader("Emission Probabilities Heatmap")
            fig = create_heatmap(emission_matrix,
                               [s.replace('_', ' ').title() for s in states],
                               [o.replace('_', ' ').title() for o in observations])
            st.plotly_chart(fig)
            
            # Model download section
            st.subheader("Download Model")
            if st.button("Download Trained Model"):
                model_data = {
                    "transition_matrix": transition_matrix.tolist(),
                    "emission_matrix": emission_matrix.tolist(),
                    "states": states,
                    "observations": observations
                }
                st.download_button(
                    label="Download as JSON",
                    data=json.dumps(model_data, indent=2),
                    file_name=f"trained_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

    else:
        st.warning("Unable to load the model. Please check your file path and model format.")

    # Add footer information
    st.sidebar.markdown("---")
    st.sidebar.write("Model last updated:", datetime.now().strftime("%Y-%m-%d"))

if __name__ == "__main__":
    main()