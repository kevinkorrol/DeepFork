from collections.abc import Generator
from pathlib import Path
import chess
import chess.pgn
import torch
from utils.chess_utils import game_to_tensors

def get_project_root() -> Path:
    """
    :return: Path to the project root
    """
    return Path(__file__).resolve().parents[1]

def load_n_processed_games(n=None) -> Generator[chess.pgn.Game, None, None]:
    """
    Loads n games from a .pt file.
    :param n: Number of games that will be put into one .pt file
    :yield: A generator of games
    """
    root = get_project_root()
    raw_dir = root / "data/raw"
    files = list(raw_dir.glob("*.pgn"))
    count = 0

    for path in files:
        with open(path, encoding="utf-8") as pgn:
            while True:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                yield game
                count += 1
                if n is not None and count >= n:
                    return

def save_all_games_in_files(n_per_file=100) -> None:
    """
    :param n_per_file: Number of games to save in each file
    """
    root = get_project_root()
    processed_dir = root / "data/processed"
    processed_dir.mkdir(exist_ok=True, parents=True)

    buffer = []
    file_idx = 0
    games_in_buffer = 0

    for game in load_n_processed_games():
        samples = game_to_tensors(game)
        buffer.extend(samples)
        games_in_buffer += 1

        if games_in_buffer >= n_per_file:
            torch.save(buffer, processed_dir / f"games_{file_idx:03d}.pt")
            file_idx += 1
            buffer = []
            games_in_buffer = 0

    if buffer:
        torch.save(buffer, processed_dir / f"games_{file_idx:03d}.pt")



if __name__ == "__main__":
    save_all_games_in_files(n_per_file=100)
