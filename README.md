# NFL Markov Predictor

## Description

This program takes in a given dataset of sports game outcomes and uses it to train a Markov Model. This is then used to predict the outcomes of sports games in a separate set of data, in order to validate that the code is working as intended.

## Installation

todo add this

## Usage (Command Line)

To run this code from the command line, run cli.py with the arguments corresponding to the processes you wish to run.
--process: Process the input data into .pkl files for training and validation, and creates the model matrices for the Markov model
--dataset: Selects the desired dataset to use for training and validation models (default is data/nfl_dataset.json)
--train: Takes in the generated training model and uses it to train the Markov model
--model: Determines where the Markov model will be output to / input from (will save at model/saves/model_\*model\*.json, defaults to model/saves/trained_model.json)
--validate: Generates accuracy metrics for the Markov model by comparing the results from it to the validation data

## Usage (App)

todo add this
