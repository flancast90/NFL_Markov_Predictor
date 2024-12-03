import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from markov import HMM

st.set_page_config(layout="wide")

# Initialize session state
if 'model' not in st.session_state:
    transition_matrix = np.zeros((4, 4))
    emission_matrix = np.zeros((2, 2))
    st.session_state.model = HMM(transition_matrix, emission_matrix)

with st.sidebar:
    st.header("Schema")
    n_sequences = st.number_input("Number of sequences", 1, 20, 5)
    sequences = []
    for i in range(n_sequences):
        seq = st.text_input(f"Sequence {i+1}", "W L W")
        sequences.append(seq.split())

# Main layout
col1, col2 = st.columns(2)

with col1:
    st.header("Model Parameters")
    
    
    st.subheader("Transition Matrix")
    fig = go.Figure(data=go.Heatmap(
        z=st.session_state.model.transition_matrix,
        text=st.session_state.model.transition_matrix,
        texttemplate="%{text:.2f}",
        textfont={"size": 10}
    ))
    st.plotly_chart(fig)
    
    st.subheader("Emission Matrix")
    fig = go.Figure(data=go.Heatmap(
        z=st.session_state.model.emission_matrix,
        text=st.session_state.model.emission_matrix,
        texttemplate="%{text:.2f}",
        textfont={"size": 10}
    ))
    st.plotly_chart(fig)

with col2:
    st.header("Training")
    if st.button("Train Model"):
        flat_sequence = [item for seq in sequences for item in seq]
        st.session_state.model.train(flat_sequence)
        st.success("Model trained successfully!")

st.markdown("---")
if sequences:
    likelihood = st.session_state.model.likelihood(sequences[0])
    st.markdown(f"### Outcome\nLikelihood of first sequence: {likelihood:.4f}")