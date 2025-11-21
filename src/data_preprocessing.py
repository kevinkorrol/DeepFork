# Võta failist n mängu (data/raw)
# Tee mängust andmestruktuur nii, et oleks state, järgmine käik ja kes võitis (iga state sama, 0 või 1)
# salvesta mäng kuidagi (data/processed) - torch dataset
#
from pathlib import Path

import chess
import chess.pgn
import torch

from utils.chess_utils import game_to_tensors

def load_n_processed_games(n) -> list:
    """
    :param n: Number of games to load
    :return: List of gamess
    """
    games = []
    root = get_project_root()
    data_dir = root / "data/raw"

    files = list(data_dir.glob("*.pgn"))
    count = 0

    for path in files:
        with open(path, encoding="utf-8") as pgn:
            game = chess.pgn.read_game(pgn)
            while game and count < n:
                games.append(game)
                count += 1
                if count >= n:
                    return games
                game = chess.pgn.read_game(pgn)

    return games


def make_training_samples(n) -> list:
    """
    :param n: Number of games to load
    :return: List of samples
    """
    games = load_n_processed_games(n)
    all_samples = []

    for game in games:
        samples = game_to_tensors(game)
        all_samples.extend(samples)

    return all_samples



def save_samples(samples) -> None:
    """
    :param samples: list of samples
    :return: Nothing, saves samples to games.pt
    """
    root = get_project_root()
    processed_dir = root / "data/processed"
    processed_dir.mkdir(exist_ok=True, parents=True)

    torch.save(samples, processed_dir / "games.pt")
    print(f"Saved {len(samples)} sample(s) to games.pt")


def get_project_root() -> Path:
    """
    :return: Path to the project root
    """
    return Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    n_games = 1

    samples = make_training_samples(n_games)
    save_samples(samples)