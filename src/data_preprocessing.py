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
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.chess_utils import game_to_tensors

def get_project_root() -> Path:
    """
    :return: Path to the project root
    """
    return Path(__file__).resolve().parents[1]

def load_n_processed_games(n, origin_dir="data/raw") -> Generator:
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


def save_all_games_in_files(samples_per_file=300, n_games=None, history_count=8) -> None:
    """
    Save processed games into chunked .pt files under data/processed.

    :param samples_per_file: Number of samples to save in each file
    :param n_games: Number of games to be loaded
    """
    root = get_project_root()
    processed_dir = root / "data/processed"
    processed_dir.mkdir(exist_ok=True, parents=True)

    buffer = []
    file_idx = 0

    for game in tqdm(load_n_processed_games(n_games),
                     desc="Processing games",
                     unit="games"):
        samples = game_to_tensors(game, history_count)
        buffer.extend(samples)

        if len(buffer) >= samples_per_file:
            torch.save(buffer[:samples_per_file], processed_dir / f"games_{file_idx:03d}.pt")
            file_idx += 1
            buffer = buffer[samples_per_file:]

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
        for game in tqdm(load_n_processed_games(None, origin_dir="data/temp"),
                         desc="Filtering Games",
                         unit="games",
                         bar_format=""):
            try:
                white_elo = game.headers.get('WhiteElo')
                black_elo = game.headers.get("BlackElo")
                if white_elo.isnumeric() and int(white_elo) > min_elo \
                        and white_elo.isnumeric() \
                        and int(black_elo) > min_elo \
                        and game.end().ply() > min_half_moves:
                    count += 1
                    print(game, file=dest, end="\n\n")
            except Exception as e:
                print(f"Skipping a corrupted game: {e}")
                continue
    print(f"Processed {count} games.")


if __name__ == "__main__":
    save_all_games_in_files(samples_per_file=300, n_games=50000, history_count=6)
