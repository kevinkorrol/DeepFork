"""
Chess-related tensor utilities and encodings for DeepFork.

Includes helpers to build input tensors from game states, maintain history,
encode global features, and map between chess moves and the 73x8x8 action space.
"""
import chess
import chess.pgn
import numpy as np
from collections.abc import Hashable

def game_to_tensors(game: chess.pgn.Game, history_count: int) -> list:
    """
    :param game: Game object to be converted
    :param history_count: Number of states to include in history
    :return: A list of sample dicts: { state, move }
    """
    current_board = game.board()
    state_history = np.zeros((history_count, 14, 8, 8), dtype=np.float32)
    seen_states = {get_state_hash(current_board): 1}
    samples = []

    for move in game.mainline_moves():
        action = move_to_action(move)
        state = state_to_tensor(state_history, current_board, seen_states, history_count)
        current_board.push(move)

        sample = {
            "state": state.astype(np.float32),
            "action": action
        }

        samples.append(sample)

    return samples


def state_to_tensor(
        state_history: np.ndarray,
        new_board: chess.Board,
        seen_states: dict,
        history_count: int
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
    legal_moves_plane = get_legal_moves_plane(new_board)

    return np.concatenate([np.stack(state_history).reshape(-1, 8, 8), global_planes, legal_moves_plane], axis=0)


def update_history(
        state_history: np.ndarray,
        new_board: chess.Board,
        history_count: int,
        seen_states: dict
) -> None:
    """
    Update state history and seen states with new board.
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
            for square in board.pieces(piece, color):
                rank = square // 8
                file = square % 8
                piece_placement[i * 6 + j, rank, file] = 1
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


def get_move_distribution(
        action_distribution: np.ndarray,
        board: chess.Board,
        temp: float = 1.0
) -> dict:
    """
    Gets distribution of moves from action distribution by mapping only legal moves from it.
    The action array represents a 73x8x8 action encoding.
    :param action_distribution:
    :param board: Current board state
    :return: Probability distribution over all legal moves
    """

    legal_moves, move_mask = get_legal_moves_mask(board)
    legal_logits = action_distribution[move_mask]

    # Normalize it
    legal_logits = (legal_logits - np.max(legal_logits)) / temp
    exp_logits = np.exp(legal_logits)
    probs = exp_logits / exp_logits.sum()

    return {move: prob for move, prob in zip(legal_moves, probs)}


def get_legal_moves_mask(board: chess.Board) -> tuple:
    legal_moves = list(board.legal_moves)
    legal_moves_idx = np.array([move_to_action(move) for move in legal_moves])
    move_mask = np.zeros(4672, dtype=bool)
    move_mask[legal_moves_idx] = True
    return legal_moves, move_mask


def get_legal_moves_plane(board: chess.Board) -> np.ndarray:
    plane = np.zeros((1, 8, 8), dtype=np.float32)
    for move in board.legal_moves:
        to_square = move.to_square
        rank = to_square // 8
        file = to_square % 8
        plane[0, rank, file] = 1.0

    return plane


DIRECTIONS = [
    (0, 1),   # N
    (1, 1),   # NE
    (1, 0),   # E
    (1, -1),  # SE
    (0, -1),  # S
    (-1, -1), # SW
    (-1, 0),  # W
    (-1, 1)   # NW
]

KNIGHT_DIFFS = [
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2)
]

PROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


def move_to_action(move: chess.Move) -> int:
    from_sq = move.from_square
    to_sq = move.to_square

    fx, fy = chess.square_file(from_sq), chess.square_rank(from_sq)
    tx, ty = chess.square_file(to_sq), chess.square_rank(to_sq)
    dx, dy = tx - fx, ty - fy

    # Queenlike moves
    if dx == 0 or dy == 0 or abs(dx) == abs(dy):
        # identify direction
        sdx = (0 if dx == 0 else (1 if dx > 0 else -1))
        sdy = (0 if dy == 0 else (1 if dy > 0 else -1))

        for dir_idx, (vx, vy) in enumerate(DIRECTIONS):
            if (vx, vy) == (sdx, sdy):
                break

        distance = max(abs(dx), abs(dy))  # 1..7
        plane = dir_idx * 7 + (distance - 1)  # 0..55

        return plane * 64 + from_sq

    # KNightlike moves
    for k, (kx, ky) in enumerate(KNIGHT_DIFFS):
        if dx == kx and dy == ky:
            plane = 56 + k  # planes 56–63
            return plane * 64 + from_sq

    # Promotions
    if move.promotion in PROMO_PIECES:
        # direction: forward / capture-left / capture-right
        if dx == 0:
            promo_dir = 0       # forward
        elif dx == -1:
            promo_dir = 1       # capture-left
        else:
            promo_dir = 2       # capture-right

        promo_piece_idx = PROMO_PIECES.index(move.promotion)

        plane = 64 + promo_dir * 3 + promo_piece_idx  # planes 64–72
        return plane * 64 + from_sq

    raise ValueError(f"Move {move} not representable in AlphaZero encoding.")

def action_to_move(action: int) -> chess.Move:
    plane = action // 64
    from_sq = action % 64
    fx, fy = chess.square_file(from_sq), chess.square_rank(from_sq)

    # Queenlike moves
    if plane < 56:
        dir_idx = plane // 7
        dist = (plane % 7) + 1

        vx, vy = DIRECTIONS[dir_idx]
        tx = fx + vx * dist
        ty = fy + vy * dist

        if 0 <= tx < 8 and 0 <= ty < 8:
            return chess.Move(
                from_sq,
                chess.square(tx, ty)
            )

    # Knightlike moves
    if 56 <= plane < 64:
        k = plane - 56
        kx, ky = KNIGHT_DIFFS[k]

        tx = fx + kx
        ty = fy + ky

        if 0 <= tx < 8 and 0 <= ty < 8:
            return chess.Move(
                from_sq,
                chess.square(tx, ty)
            )

    # Promotions
    if 64 <= plane < 73:
        sub = plane - 64

        promo_dir = sub // 3  # 0..2
        promo_piece_idx = sub % 3
        promo_piece = PROMO_PIECES[promo_piece_idx]

        dx = [0, -1, 1][promo_dir]
        dy = 1 if chess.square_rank(from_sq) == 6 else -1  # white/black

        tx = fx + dx
        ty = fy + dy

        if 0 <= tx < 8 and 0 <= ty < 8:
            return chess.Move(
                from_sq,
                chess.square(tx, ty),
                promotion=promo_piece
            )

    raise ValueError(f"Action {action} out of range.")


if __name__ == "__main__":
    states = {get_state_hash(chess.Board()): 1}
    history_count = 1
    history = np.zeros((history_count, 14, 8, 8), dtype=np.float32)

    example_board = chess.Board()
    state_to_tensor(history, example_board, states, history_count)
    example_board.push_san("Nf3")
    state_to_tensor(history, example_board, states, history_count)
    example_board.push_san("Nc6")
    state_to_tensor(history, example_board, states, history_count)
    example_board.push_san("Ng1")
    state_to_tensor(history, example_board, states, history_count)
    example_board.push_san("Nb8")
    np.set_printoptions(threshold=np.inf)

    print("Legal moves:")
    for m in list(example_board.legal_moves):
        print(m, move_to_action(m))

    # Example random action distribution
    raw_distribution = np.random.rand(4672).astype(np.float32)
    legal_dist = get_move_distribution(raw_distribution, example_board)

    print("\nLegal move distribution:")
    print(legal_dist)

    print("\nSum of legal distribution:", sum(legal_dist.values()))
    print("Number of legal moves:", len(legal_dist))

    move = chess.Move(chess.B1, chess.B6)
    action = move_to_action(move)
    print(action)
    print(action_to_move(action))

    #print(state_to_tensor(history, example_board, states, history_count))