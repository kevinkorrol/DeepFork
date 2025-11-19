import chess
import numpy as np
from collections.abc import Hashable


def state_to_tensor(boards: list, seen_states: dict) -> np.ndarray:
    """
    Return a 119x8x8 tensor of board state where
      - planes 1-112 are the last 8 piece placements and their
        repetition counter planes (12 for piece placement and 2 repetition counter planes each)
      - planes 113-119 are global features.
    :param boards: A list of the last 8 board states (if less than 8, fill with empty boards)
    :param seen_states: A dictionary of seen state, repetition count pairs
    :return: 119x8x8 tensor of board state
    """
    state_planes = np.zeros((119, 8, 8), dtype=np.float32)

    boards.reverse()
    for i, board in enumerate(boards):
        piece_placement = get_piece_placement_planes(board)
        repetition_planes = get_repetition_counter_planes(board, seen_states)
        state_planes[i * 14:(i + 1) * 14, :, :] = np.concatenate([piece_placement, repetition_planes])
    state_planes[112:, :, :] = get_global_planes(boards[0])

    return state_planes


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


def get_repetition_counter_planes(board: chess.Board, seen_states: dict) -> np.ndarray:
    """
    Return 2x8x8 tensor of repetition counter.
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

    example_board = chess.Board()
    example_board.push_san("Nf3")
    example_board.push_san("Nc6")
    state_to_tensor([example_board], states)
    example_board.push_san("Ng1")
    example_board.push_san("Nb8")
    print(state_to_tensor([example_board], states).shape)
    state_to_tensor([example_board], states)