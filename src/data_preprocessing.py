# Võta failist n mängu (data/raw)
# Tee mängust andmestruktuur nii, et oleks state, järgmine käik ja kes võitis (iga state sama, 0 või 1)
# salvesta mäng kuidagi (data/processed) - torch dataset
#
from pathlib import Path

import chess
import chess.pgn
import torch

from utils.chess_utils import game_to_tensors


def raw_game_to_ready_tensor(game : chess.pgn.Game):


    tensors = game_to_tensors(game) ## returns dict, maybe should return something else
    # tensors will be inserted into model using ?dataloader? (neural network) and the models output can be compared to made move and winner

    return tensors


def load_n_processed_games(n):
    games = []
    root = Path.cwd().parent
    data_dir = root / "data/raw"

    files = list(data_dir.glob("*.pgn"))
    count = 0

    for path in files:
        with open(path, encoding="utf-8") as pgn:
            while count < n:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                games.append(game)
                count += 1
        if count >= n:
            break

    return games


def make_training_samples(n):
    games = load_n_processed_games(n)
    all_samples = []

    for game in games:
        samples = game_to_tensors(game)
        all_samples.append(samples)

    return all_samples



def save_samples(samples):
    processed_dir = Path.cwd().parent / "data/processed"
    processed_dir.mkdir(exist_ok=True, parents=True)

    torch.save(samples, processed_dir / "games.pt")
    print(f"Saved {len(samples)} samples to games.pt")


if __name__ == '__main__':
    n_games = 1

    samples = make_training_samples(n_games)
    save_samples(samples)