# DeepFork - Neural net that sees every fork coming

Repo authors: Tõnis Lees, Kevin Korrol

## Project motivation and goal

The goal of this project was to learn about chess engines and create and train a model that would make better moves than random. We decided that the project would be a success if our chess engine can beat the lowest rated bot on chess.com website, called Martin. In case the model works really well, it could be strong enough to beat an average player easily. 


## Repo guide

### / -

get_data scripts are used to get all the games from the pgnmentor.com page and combine them into one processed .pgn file


### /data - 

folder for storing .pgn and .pt files 

### /models - 

folder for storing trained models

### /notebooks - 

.ipynb files that are not really part of the whole project flow, but were used to first analyse the capabilites of the chess library and .pgn files. Towards the end the notebook was used to analyse the training data

### /src - 

General folder containing files used to play chess against a trained model, train the models, ...

### /utils - 

Contains files with helper functions for the whole workflow


## Replicating results

The first step would be to fetch the .png data from the internet using the get_data script

After that, the .pgn files need to be converted into .pt files that can be used as input for the model

When that is done, run train.py to train the model.




HW10: https://docs.google.com/document/d/12dOD-GB6gpn7qroGx5NYyiwk3bp99Q19QCCgOZOlU38/edit?usp=sharing
