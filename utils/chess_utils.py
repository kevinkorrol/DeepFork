import chess
import numpy as np

def state_to_tensor(boards: list) -> np.ndarray:
    """
    Return a 119x8x8 tensor of board state where
    - planes 1-112 are the last 8 piece placements and their
      repetition counter planes (12 for piece placement and 2 repetition counter planes each)
    - planes 113-119 are global features.
    """
    state_planes = np.zeros((119, 8, 8), dtype=np.float32)

    boards.reverse()
    for i, board in enumerate(boards):
        piece_placement = get_piece_placement_planes(board)
        repetition_planes = get_repetition_counter_planes(board)
        state_planes[i * 14:(i + 1) * 14, :, :] = np.stack([piece_placement, repetition_planes])
    state_planes[112:, :, :] = get_global_planes(boards[0])

    return state_planes


def get_piece_placement_planes(board: chess.Board) -> np.ndarray:
    """Return 12x8x8 tensor of piece placement. One plane for each piece type for both players."""
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


def get_repetition_counter_planes(board: chess.Board) -> np.ndarray:
    """
    Return 2x8x8 tensor of repetition counter.
    If 0 repetitions, return all zeros.
    If 1 repetition, return plane_1 = ones, plane_2 = zeros.
    If 2 or more repetitions, return plane_1 = zeros, plane_2 = ones.
    """
    repetition_counter_planes = np.zeros((2, 8, 8), dtype=np.float32)

    # Board has been seen before two or more times
    if board.is_repetition(3):
        repetition_counter_planes[1, :, :] = 1.0
    # Board has been seen before once
    elif board.is_repetition(2):
        repetition_counter_planes[0, :, :] = 1.0

    return repetition_counter_planes


def get_global_planes(board: chess.Board) -> np.ndarray:
    global_planes = np.zeros((7, 8, 8), dtype=np.float32)
    # TODO
    return global_planes


if __name__ == "__main__":
    example_board = chess.Board()
    example_board.push_san("Nf3")
    example_board.push_san("Nc6")
    print(get_piece_placement_planes(example_board))
    print(get_repetition_counter_planes(example_board))
#    print(get_global_planes(board))