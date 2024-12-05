# NFL Markov Predictor

## Description

This program takes in a given dataset of sports game outcomes and uses it to train a Markov Model. This is then used to predict the outcomes of sports games in a separate set of data, in order to validate that the code is working as intended.

## Installation

Download the file, as well as Python 3.12.7. In addition, you will need to install the following libraries, either to PATH or to the same folder as your program:
* Streamlit
* Graphviz
* Numpy
* Pandas
* Plotly

## Usage (Command Line)

To run this code from the command line, run cli.py with the arguments corresponding to the processes you wish to run.
* --process: Process the input data into .pkl files for training and validation, and creates the model matrices for the Markov model
* --dataset: Selects the desired dataset to use for training and validation models (default is data/nfl_dataset.json)
* --train: Takes in the generated training model and uses it to train the Markov model
* --model: Determines where the Markov model will be output to / input from (will save at model/saves/model_\*model\*.json, defaults to model/saves/trained_model.json)
* --validate: Generates accuracy metrics for the Markov model by comparing the results from it to the validation data
* --montecarlo <its>: Runs its iterations of the Monte Carlo simulation.

## Usage (App)

To run this code as an app in your web browser, run the command "streamlit run app.py" from the command line. From here, select the desired tab on the left.
* Model Visualization: View a visualization of the displayed model, as well as the transition and emission matrices.
* Make Predictions: Predict which of two teams will win based on team history, and view the recent history of matchups between the two selected teams.
* Model Analysis: View the transition and emission matrices in the forms of heatmaps, and download the trained model data.
