# DeepFork

Repo authors: Tõnis Lees, Kevin Korrol

## Project motivation and goal

The goal of this project was to learn about chess engines and create and train a model that would make better moves than random. We decided that the project would be a success if our chess engine can beat the lowest rated bot on chess.com website, called Martin. In case the model works really well, it could be strong enough to beat an average player easily. 


## Repo Guide

### `/`

`get_data` scripts are used to get all the games from the pgnmentor.com page and combine them into one processed `.pgn` file.

### `/data`

Folder for storing `.pgn` and `.pt` files:  
- `/processed` - `.pt` files to store input tensors  
- `/raw` - `.pgn` files for raw unprocessed data

### `/models`

Folder for storing trained models.

### `/notebooks`

`.ipynb` files that are not part of the main project flow but were used to analyze the the training data.

### `/src`

General folder containing files used to play chess against a trained model, train models, etc.

### `/utils`

Contains helper functions for the whole workflow.



## Replicating results

The first step would be to fetch the .png data from the internet using the get_data script.
After that, the .pgn files need to be converted into .pt files that can be used as input for the model.
When that is done, run train.py to train the model.




HW10: https://docs.google.com/document/d/12dOD-GB6gpn7qroGx5NYyiwk3bp99Q19QCCgOZOlU38/edit?usp=sharing
