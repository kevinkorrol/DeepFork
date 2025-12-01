"""
Data preprocessing utilities for converting PGN games into training tensors.

Includes helpers to locate project paths, stream chess games from PGN files,
and serialize processed samples to torch .pt files in fixed-size shards.
"""

from collections.abc import Generator
from pathlib import Path
import chess
import chess.pgn
import torch
from tqdm import tqdm
from utils.chess_utils import game_to_tensors

def get_project_root() -> Path:
    """
    :return: Path to the project root
    """
    return Path(__file__).resolve().parents[1]

def load_n_processed_games(n=None, origin_dir="data/raw") -> Generator[chess.pgn.Game, None, None]:
    """
    Stream up to n chess games from PGN files under data/raw.
    :param n: Maximum number of games to yield (None for all)
    :param origin_dir: A directory that is read
    :yield: A generator of games
    """
    root = get_project_root()
    raw_dir = root / origin_dir
    files = list(raw_dir.glob("*.pgn"))
    count = 0

    for path in files:
        with open(path, encoding="utf-8") as pgn:
            while True:
                try:
                    game = chess.pgn.read_game(pgn)
                except Exception as e:
                    print(f"Skipping a corrupted game in file {path.name}: {e}")
                    continue
                if game is None:
                    break
                yield game
                count += 1
                if n is not None and count >= n:
                    return

def save_all_games_in_files(n_per_file=100) -> None:
    """
    Save processed games into chunked .pt files under data/processed.

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


def filter_games(min_elo: int = 2400, min_half_moves: int = 30) -> None:
    """
    Filters all games in /temp to be only with ratings over 2400 and writes them to dataset.pgn.
    """
    root = get_project_root()
    dataset_path = root / "data/raw/dataset.pgn"
    count = 0

    with open(dataset_path, 'w') as dest:
        for game in tqdm(load_n_processed_games(origin_dir="data/temp"),
                         desc="Filtering Games",
                         unit="games",
                         bar_format=""):
            white_elo = game.headers.get('WhiteElo')
            black_elo = game.headers.get("BlackElo")
            if white_elo and int(white_elo) > min_elo \
                    and black_elo \
                    and int(black_elo) > min_elo \
                    and game.end().ply() > min_half_moves:
                count += 1
                print(game, file=dest, end="\n\n")
    print(f"Processed {count} games.")


if __name__ == "__main__":
    save_all_games_in_files(n_per_file=100)
