import streamlit as st
import json
from graphviz import Digraph
import numpy as np
import pandas as pd
import os
import plotly.express as px
from datetime import datetime
from typing import List, Dict, Tuple
from functools import lru_cache
from model.validate import ModelValidator

# Configure page and styling
st.set_page_config(
    page_title="HMM Sports Predictor", layout="wide", initial_sidebar_state="expanded"
)

# Load README content
try:
    with open("README.md", "r") as f:
        readme_content = f.read()
except Exception as e:
    readme_content = "Error loading README.md"

# Custom CSS for better styling
st.markdown(
    """
<style>
    .stApp {
        margin: 0 auto;
    }
    .main > div {
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #1f77b4;
        font-size: 2.5rem !important;
        margin-bottom: 2rem !important;
    }
    h2 {
        color: #2c3e50;
        font-size: 1.8rem !important;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        background-color: #1f77b4;
        color: white;
    }
    .stSelectbox {
        margin-bottom: 1rem;
    }
    .css-1d391kg {  /* Sidebar styling */
        background-color: #f8f9fa;
    }
    .graphviz-chart {
        margin: 0 auto;
        width: 100%;
    }
    @media (min-width: 1200px) {
        .graphviz-chart {
            width: 75%;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)


@lru_cache(maxsize=1)
def load_hmm_model_from_file(file_path):
    try:
        if not os.path.exists(file_path):
            st.error(f"Could not find file: {file_path}")
            return None, None, None, None

        with open(file_path, "r") as file:
            model_data = json.load(file)

        transition_matrix = np.exp(np.array(model_data["transition_matrix"])[1:3, 1:3])
        emission_matrix = np.exp(np.array(model_data["emission_matrix"]))

        return (
            transition_matrix,
            emission_matrix,
            model_data["states"],
            model_data["observations"],
        )

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None


@st.cache_data
def create_hmm_visualization(transition_matrix, emission_matrix, states, observations):
    dot = Digraph(comment="Hidden Markov Model", engine="dot")
    dot.attr(rankdir="LR", bgcolor="transparent", size="1,1")  # Half width

    # Enhanced node styling
    node_styles = {
        "state": {
            "shape": "circle",
            "style": "filled",
            "fillcolor": "#1f77b4",
            "fontcolor": "white",
            "width": "0.75",  # Half width
            "penwidth": "1",  # Half width
        },
        "obs": {
            "shape": "box",
            "style": "filled",
            "fillcolor": "#ff7f0e",
            "fontcolor": "white",
            "width": "0.75",  # Half width
            "penwidth": "1",  # Half width
        },
    }

    with dot.subgraph(name="cluster_states") as c:
        c.attr(rank="same")
        for state in states:
            c.node(
                f"state_{state}",
                state.replace("_", " ").title(),
                **node_styles["state"],
            )

    with dot.subgraph(name="cluster_observations") as c:
        c.attr(rank="same")
        for obs in observations:
            c.node(f"obs_{obs}", obs.replace("_", " ").title(), **node_styles["obs"])

    # Enhanced edge styling
    for i, from_state in enumerate(states):
        for j, to_state in enumerate(states):
            prob = transition_matrix[i][j]
            if prob > 0.01:
                dot.edge(
                    f"state_{from_state}",
                    f"state_{to_state}",
                    label=f"{prob*100:.1f}%",
                    penwidth=str(0.5 + 1.5 * prob),  # Half width
                    color="#1f77b4",
                    fontcolor="#2c3e50",
                )

    for i, state in enumerate(states):
        for j, obs in enumerate(observations):
            prob = emission_matrix[j][i]
            if prob > 0.01:
                dot.edge(
                    f"state_{state}",
                    f"obs_{obs}",
                    label=f"{prob*100:.1f}%",
                    style="dashed",
                    penwidth="0.75",  # Half width
                    color="#ff7f0e",
                    fontcolor="#2c3e50",
                )

    return dot


@st.cache_data
def create_heatmap(matrix, labels_x, labels_y):
    fig = px.imshow(
        matrix,
        labels=dict(x="To State", y="From State", color="Probability"),
        x=labels_x,
        y=labels_y,
        color_continuous_scale="RdYlBu_r",
        aspect="auto",
    )
    fig.update_layout(
        font_family="Arial",
        font_size=14,
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


@st.cache_data
def load_nfl_games(file_path: str) -> List[Dict]:
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except Exception as e:
        st.error(f"Error loading NFL games data: {str(e)}")
        return []


@st.cache_data
def get_unique_teams(games: List[Dict]) -> List[str]:
    return sorted(
        list(
            {game["home_team"] for game in games}
            | {game["away_team"] for game in games}
        )
    )


def predict_winner(home_team, away_team):
    validator = ModelValidator()
    validator.load_model()
    validator.load_validation_data()
    prediction, confidence = validator.predict(
        home_team, away_team, provide_confidence=True
    )

    probabilities = {
        home_team: confidence if prediction == "home_win" else 0,
        away_team: confidence if prediction == "away_win" else 0,
    }

    winner = home_team if prediction == "home_win" else away_team
    return winner, probabilities


@st.cache_data
def get_team_history(games: List[Dict], team: str) -> Tuple[int, int]:
    wins = losses = 0

    for game in games:
        try:
            home_score = game.get("home_score")
            away_score = game.get("away_score")

            if not (home_score and away_score):
                continue

            is_home = game["home_team"] == team
            is_away = game["away_team"] == team
            home_won = int(home_score) > int(away_score)
            away_won = not home_won

            if is_home:
                wins += home_won
                losses += not home_won
            elif is_away:
                wins += away_won
                losses += not away_won

        except (TypeError, ValueError):
            continue

    return wins, losses


def display_matchup_history(games: List[Dict], home_team: str, away_team: str):
    matchups = [
        g for g in games if {g["home_team"], g["away_team"]} == {home_team, away_team}
    ]

    if matchups:
        st.markdown(
            """
        <style>
        .matchup-table {
            font-family: Arial, sans-serif;
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }
        .matchup-table th {
            color: white;
            padding: 12px;
            text-align: left;
        }
        .matchup-table td {
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        matchup_data = []
        for game in matchups[-5:]:
            try:
                home_score = game.get("home_score", "N/A")
                away_score = game.get("away_score", "N/A")

                if home_score != "N/A" and away_score != "N/A":
                    winner = (
                        game["home_team"]
                        if int(home_score) > int(away_score)
                        else game["away_team"]
                    )
                else:
                    winner = "Unknown"

                matchup_data.append(
                    {
                        "Date": str(game["date"]),
                        "Home": f"{game['home_team']} ({home_score})",
                        "Away": f"{game['away_team']} ({away_score})",
                        "Winner": winner,
                    }
                )
            except (TypeError, ValueError):
                continue

        if matchup_data:
            df = pd.DataFrame(matchup_data)
            st.table(df)
        else:
            st.info("No valid matchup data found between these teams")
    else:
        st.info("No recent matchups found between these teams")


def main():
    st.sidebar.title("🏈 Navigation")
    page = st.sidebar.radio(
        "",
        ["Documentation", "Model Visualization", "Make Predictions", "Model Analysis"],
        format_func=lambda x: (
            "📚 Documentation"
            if x == "Documentation"
            else (
                f"📊 {x}"
                if x == "Model Analysis"
                else f"🔮 {x}" if x == "Make Predictions" else f"📈 {x}"
            )
        ),
    )

    if page == "Documentation":
        st.markdown(readme_content)
        return

    file_path = "model\\saves\\trained_model.json"
    transition_matrix, emission_matrix, states, observations = load_hmm_model_from_file(
        file_path
    )

    if all(
        v is not None
        for v in [transition_matrix, emission_matrix, states, observations]
    ):
        if page == "Model Visualization":
            st.title("Hidden Markov Model Visualization")

            with st.expander("ℹ️ About this visualization", expanded=True):
                st.markdown(
                    """
                This interactive visualization shows the structure of the Hidden Markov Model:
                - 🔵 **States** (blue circles): Hidden states representing game outcomes
                - 🟠 **Observations** (orange boxes): Observable game results
                - ➡️ **Solid arrows**: Transition probabilities between states
                - ➡️ **Dashed arrows**: Emission probabilities from states to observations
                """
                )

            dot = create_hmm_visualization(
                transition_matrix, emission_matrix, states, observations
            )
            st.markdown('<div class="graphviz-chart">', unsafe_allow_html=True)
            st.graphviz_chart(dot, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                with st.expander("📊 Transition Probabilities"):
                    transition_df = pd.DataFrame(
                        transition_matrix * 100,
                        columns=[s.replace("_", " ").title() for s in states],
                        index=[s.replace("_", " ").title() for s in states],
                    )
                    st.dataframe(
                        transition_df.style.background_gradient(cmap="Blues").format(
                            "{:.1f}%"
                        ),
                        use_container_width=True,
                    )

            with col2:
                with st.expander("📊 Emission Probabilities"):
                    emission_df = pd.DataFrame(
                        emission_matrix * 100,
                        columns=[s.replace("_", " ").title() for s in states],
                        index=[o.replace("_", " ").title() for o in observations],
                    )
                    st.dataframe(
                        emission_df.style.background_gradient(cmap="Oranges").format(
                            "{:.1f}%"
                        ),
                        use_container_width=True,
                    )

        elif page == "Make Predictions":
            st.title("🏈 Game Outcome Predictor")

            games = load_nfl_games("data\\nfl_dataset.json")

            if games:
                teams = get_unique_teams(games)

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("🏠 Home Team")
                    home_team = st.selectbox(
                        "Select Home Team",
                        teams,
                        key="home_team",
                        on_change=lambda: st.session_state.pop("prediction", None),
                    )

                    if home_team:
                        wins, losses = get_team_history(games, home_team)
                        win_pct = (
                            (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
                        )
                        st.metric(
                            "Season Record",
                            f"{wins}-{losses}",
                            f"{win_pct:.1f}% Win Rate",
                        )

                with col2:
                    st.subheader("✈️ Away Team")
                    away_teams = [team for team in teams if team != home_team]
                    away_team = st.selectbox(
                        "Select Away Team",
                        away_teams,
                        key="away_team",
                        on_change=lambda: st.session_state.pop("prediction", None),
                    )

                    if away_team:
                        wins, losses = get_team_history(games, away_team)
                        win_pct = (
                            (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
                        )
                        st.metric(
                            "Season Record",
                            f"{wins}-{losses}",
                            f"{win_pct:.1f}% Win Rate",
                        )

                if home_team and away_team:
                    if "prediction" not in st.session_state:
                        predicted_winner, probabilities = predict_winner(
                            home_team, away_team
                        )
                        st.session_state.prediction = (predicted_winner, probabilities)
                    else:
                        predicted_winner, probabilities = st.session_state.prediction

                    win_probability = max(probabilities.values())

                    st.markdown("---")

                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.markdown(
                            f"""
                            <div style='
                                background-color: {'#e6ffe6' if predicted_winner == home_team else '#ffe6e6'};
                                padding: 10px;
                                border-radius: 10px;
                                text-align: center;
                                margin: 20px 0;
                            '>
                                <h2 style='margin:0; color: {'#006600' if predicted_winner == home_team else '#660000'};'>
                                    Predicted Winner: {predicted_winner}
                                </h2>
                                <p style='font-size: 1.2em; margin: 10px 0; color: {'#006600' if predicted_winner == home_team else '#660000'};'>
                                    Confidence: {win_probability:.1f}%
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.markdown("---")

                    st.subheader("📜 Recent Matchup History")
                    display_matchup_history(games, home_team, away_team)

        elif page == "Model Analysis":
            st.title("📊 Model Analysis Dashboard")

            col1, col2 = st.columns(2)

            with col1:
                with st.expander("State Transition Heatmap", expanded=True):
                    fig = create_heatmap(
                        transition_matrix,
                        [s.replace("_", " ").title() for s in states],
                        [s.replace("_", " ").title() for s in states],
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                with st.expander("Emission Probabilities Heatmap", expanded=True):
                    fig = create_heatmap(
                        emission_matrix,
                        [s.replace("_", " ").title() for s in states],
                        [o.replace("_", " ").title() for o in observations],
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with st.expander("💾 Download Model"):
                if st.button("Prepare Download"):
                    model_data = {
                        "transition_matrix": transition_matrix.tolist(),
                        "emission_matrix": emission_matrix.tolist(),
                        "states": states,
                        "observations": observations,
                    }
                    st.download_button(
                        label="📥 Download Model as JSON",
                        data=json.dumps(model_data, indent=2),
                        file_name=f"trained_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                    )

    else:
        st.error(
            "Unable to load the model. Please check your file path and model format."
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"<div style='text-align: center; color: #666;'>"
        f"Last Updated: {datetime.now().strftime('%Y-%m-%d')}"
        f"</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
