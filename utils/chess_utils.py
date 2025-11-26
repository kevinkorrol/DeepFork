import chess
import chess.pgn
import numpy as np
from collections.abc import Hashable

def game_to_tensors(game: chess.pgn.Game, history_count: int = 8) -> list:
    """
    :param game: Game object to be converted
    :param history_count: Number of states to include in history
    :return: A list of sample dicts: { state, move, result }
    """
    current_board = game.board()
    state_history = np.zeros((history_count, 14, 8, 8), dtype=np.float32)
    seen_states = {get_state_hash(current_board): 1}
    samples = []

    # result: 1 = white, 0 = draw, -1 = black
    result_str = game.headers.get("Result", "*")
    if result_str == "1-0":
        result = 1
    elif result_str == "0-1":
        result = -1
    else:
        result = 0

    for move in game.mainline_moves():
        current_board.push(move)
        state = state_to_tensor(state_history, current_board, seen_states)

        sample = {
            "state": state.astype(np.float32),  # shape (119,8,8)
            "move": move.uci(),                 # store as UCI string for now
            "result": result                    # 1, 0 or -1
        }

        samples.append(sample)

    return samples


def state_to_tensor(
        state_history: np.ndarray,
        new_board: chess.Board,
        seen_states: dict[Hashable, int],
        history_count: int=8
) -> np.ndarray:
    """
    Update state_history and return a (h*14 + 7)x8x8 tensor of board state where
      - planes 1-(h*14) are the last h piece placements and their
        repetition counter-planes (12 for piece placement and 2 repetition counter planes each)
      - planes (h*14 + 1)-(h*14 + 8) are global features.
    :param state_history: A list of last h piece placement tensors
    :param new_board: Current state
    :param seen_states: Seen state, repetition count pairs
    :param history_count: History state count
    :return: 119x8x8 tensor of board state
    """

    update_history(state_history, new_board, history_count, seen_states)
    global_planes = get_global_planes(new_board)

    return np.concatenate([np.stack(state_history).reshape(-1, 8, 8), global_planes], axis=0)


def update_history(
        state_history: np.ndarray,
        new_board: chess.Board,
        history_count: int,
        seen_states: dict[Hashable, int]
) -> None:
    """
    Update state history and seen states with new bpard.
    :param state_history: A list of last h piece placement tensors
    :param new_board: Current state
    :param history_count: History state count
    :param seen_states: Seen state, repetition count pairs
    """
    # Move all previous states by one
    for i in range(history_count - 1, 0, -1):
        state_history[i] = state_history[i - 1]

    state_history[0] = np.concatenate([
        get_repetition_counter_planes(new_board, seen_states),
        get_piece_placement_planes(new_board)
    ])


def get_piece_placement_planes(board: chess.Board) -> np.ndarray:
    """
    Return 12x8x8 tensor of piece placement. One plane for each piece type for both players.
    :param board: Board object of current state
    :return: 12x8x8 tensor of piece placement
    """
    colors = [chess.WHITE, chess.BLACK]
    pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

    piece_placement = np.zeros((12, 8, 8), dtype=np.float32)

    for i, color in enumerate(colors):
        for j, piece in enumerate(pieces):
            piece_int = board.pieces(piece, color)
            for k in range(64):
                if piece_int == 0:
                    break
                if piece_int & 1:
                    rank = k // 8
                    piece_placement[i * 6 + j, rank, k - rank * 8] = 1
                piece_int >>= 1
    return piece_placement


def get_state_hash(board: chess.Board) -> Hashable:
    """
    Create a hashable representation of the board state.
    :param board: Board object of current state
    :return: Hashable representation of board state
    """
    return (
        board.pawns,
        board.knights,
        board.bishops,
        board.rooks,
        board.queens,
        board.kings,
        board.turn,
        board.clean_castling_rights(),
        board.ep_square if board.has_legal_en_passant() else None,
    )


def get_repetition_counter_planes(board: chess.Board, seen_states: dict[Hashable, int]) -> np.ndarray:
    """
    Return 2x8x8 tensor of repetition counter and update seen_states.
    If 0 repetitions, return all zeros.
    If 1 repetition, return plane 1 = ones, plane 2 = zeros.
    If 2 or more repetitions, return plane 1 = zeros, plane 2 = ones.
    :param board: Board object of current state
    :param seen_states: A dictionary of seen state, repetition count pairs
    :return: 2x8x8 tensor of repetition counter
    """
    repetition_counter_planes = np.zeros((2, 8, 8), dtype=np.float32)

    state_hash = get_state_hash(board)
    state_count = seen_states.get(state_hash, 0)
    seen_states[state_hash] = state_count + 1

    if state_count >= 2:
        repetition_counter_planes[1, :, :] = 1.0
    elif state_count == 1:
        repetition_counter_planes[0, :, :] = 1.0

    return repetition_counter_planes


def get_global_planes(board: chess.Board) -> np.ndarray:
    """
    Create 7x8x8 tensor of global features where
      - plane 0 is 1 if it's white's turn, 0 otherwise
      - planes 1-4 are 1 if that player on queen-/kingside has castling rights, 0 otherwise
      - plane 5 is the move count normalized to be between 0 and 1 (assuming maximum of 200 moves per game)
      - plane 6 is the fifty-move rule clock normalized to be between 0 and 1
    :param board: Board object of current state
    :return: 7x8x8 tensor of global features
    """
    global_planes = np.zeros((7, 8, 8), dtype=np.float32)

    global_planes[0, :, :] = 1 if board.turn == chess.WHITE else 0

    global_planes[1, :, :] = board.has_kingside_castling_rights(chess.WHITE)
    global_planes[2, :, :] = board.has_queenside_castling_rights(chess.WHITE)
    global_planes[3, :, :] = board.has_kingside_castling_rights(chess.BLACK)
    global_planes[4, :, :] = board.has_queenside_castling_rights(chess.BLACK)

    move_count = board.fullmove_number
    global_planes[5, :, :] = move_count / 200 # Normalize with 200, since games rarely exceed 200 moves

    move_count_50 = board.halfmove_clock
    global_planes[6, :, :] = move_count_50 / 50

    return global_planes


if __name__ == "__main__":
    states = {get_state_hash(chess.Board()): 1}
    history = np.zeros((8, 14, 8, 8), dtype=np.float32)

    example_board = chess.Board()
    state_to_tensor(history, example_board, states)
    example_board.push_san("Nf3")
    state_to_tensor(history, example_board, states)
    example_board.push_san("Nc6")
    state_to_tensor(history, example_board, states)
    example_board.push_san("Ng1")
    state_to_tensor(history, example_board, states)
    example_board.push_san("Nb8")
    np.set_printoptions(threshold=np.inf)

    print(state_to_tensor(history, example_board, states))