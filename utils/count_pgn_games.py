from pathlib import Path
import sys


def count_games_in_pgn(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("[Event"):
                count += 1
    return count


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        pgn_path = Path(sys.argv[1]).resolve()
    else:
        pgn_path = get_project_root() / "data" / "raw" / "dataset.pgn"

    if not pgn_path.exists():
        print(f"PGN file not found: {pgn_path}")
        sys.exit(1)

    games = count_games_in_pgn(pgn_path)
    print(f"Total games in {pgn_path}: {games}")
