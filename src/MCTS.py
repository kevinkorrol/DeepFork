"""
Monte Carlo Tree Search (MCTS) implementation for the DeepFork chess agent.

This module defines the MCTSNode structure and the MCTS search function that
leverages policy and value estimates from the DeepForkNet model to guide search.
"""

from __future__ import annotations
from utils.chess_utils import state_to_tensor, update_history, get_move_distribution
from model import DeepForkNet
from utils.MCTS_visualization import visualize_mcts_graph

import math
import torch
import chess
import numpy as np


class MCTSNode:
    """
    A single node in the MCTS search tree representing a chess position.

    Tracks prior estimate, visit count, total value, and children, and
    provides utilities for PUCT-based selection and backpropagation.
    """
    def __init__(
            self,
            board: chess.Board,
            seen_states: dict,
            state_history: np.ndarray,
            prior_est: np.float32 = 0.0,
            parent: MCTSNode = None,
            move: chess.Move = None
    ):
        self.board = board
        self.seen_states = seen_states
        self.state_history = state_history
        self.parent = parent
        self.move = move
        self.is_expanded = False
        self.children = {}
        self.prior_est = prior_est
        self.visit_count = 0
        self.total_value = 0.0


    def is_terminal(self) -> bool:
        """
        Determines if the game state is terminal.
        :return: True if the game state is terminal, False otherwise
        """
        return self.board.is_game_over() or not list(self.board.legal_moves)


    def Q(self) -> float:
        """
        Calculates the exploitation term value for the PUCT.
        :return: Exploitation term
        """
        return 0 if self.visit_count == 0 else self.total_value / self.visit_count


    def U(self, c_puct: float, child_prier_est: np.float32, child_visit_count: int = 0) -> np.float32:
        """
        Calculates the exploration term for the PUCT.
        :param child_prier_est: Prior estimation for this state to be reached from the parent node
        :param child_visit_count: Visit count of the child that the term is calculated with
        :param c_puct: A constant controlling exploration vs. exploitation
        :return: Exploration term
        """
        return c_puct * child_prier_est * math.sqrt(self.visit_count) / (child_visit_count + 1)


    def get_best_move(self, c_puct: float) -> chess.Move:
        """
        Gets best child from all children using the PUCT formula.
        :param c_puct: A constant controlling exploration vs. exploitation
        :return: Child with the best PUCT value
        """
        best_move = None
        best_value = -math.inf
        for move, values in self.children.items():
            child, est = values
            child_visit_count = 0 if child is None else child.visit_count
            Q_value = child.Q() if child is not None else 0.0

            U_value = self.U(c_puct, est, child_visit_count)

            puct = Q_value + U_value
            if best_value < puct:
                best_move = move
                best_value = puct
        return best_move


    def add_or_get_child(self, move: chess.Move, history_count: int) -> MCTSNode:
        child, est = self.children[move]
        if child is None:
            board_copy = self.board.copy()
            board_copy.push(move)
            state_history_copy = np.copy(self.state_history)
            seen_states_copy = dict(self.seen_states)
            update_history(state_history_copy, board_copy, history_count, seen_states_copy)

            child = MCTSNode(
                board_copy,
                seen_states_copy,
                state_history_copy,
                est,
                parent=self,
                move=move
            )
            self.children[move][0] = child
        return child


    def expand(self, move_distr: dict) -> None:
        """
        Expand the current node by creating all of its children objects.
        :param move_distr: Policy head estimations from the model
        """
        self.is_expanded = True
        self.children = {move: [None, est] for move, est in move_distr.items()}


    def add_dirichlet_noise(self, prior_ests: dict):
        # TODO RL
        return {
            idx: 0.75 * est + 0.25 * np.random.dirichlet(np.zeros([len(prior_ests)], dtype=np.float32) + 0.3)
            for idx, est in prior_ests.items()
        }


    def backprop(self, value: float):
        """
        Adds values to all the nodes in the path from root to the current node.
        :param value: Value from models value head
        """
        node = self
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            node = node.parent
            value = -value # Flip value for the other player


    def select_best_child(self) -> chess.Move:
        """
        Select the move that leads to the most visited child from the root.

        :return: Move associated with the child with highest visit count
        """
        best_move = None
        best_count = -1
        for move, (child, _) in self.children.items():
            current_count = child.visit_count if child is not None else 0
            if child is not None and current_count > best_count:
                best_count = current_count
                best_move = move
        return best_move

    def evaluate_board_simple(self, board: chess.Board) -> float:
        """
        Evaluates the board score (White Advantage - Black Advantage).
        Uses simplified Piece-Square Tables for positional awareness.
        """
        # Tables from PeSTO / Simplified evaluation logic
        # Values are for White; mirror for Black.
        # Tables are 1D arrays of 64 squares (a1..h1, a2..h2...)

        # Pawn Table (Encourages center control and advancement)
        PAWN_PST = [
            0, 0, 0, 0, 0, 0, 0, 0,
            50, 50, 50, 50, 50, 50, 50, 50,
            10, 10, 20, 30, 30, 20, 10, 10,
            5, 5, 10, 25, 25, 10, 5, 5,
            0, 0, 0, 20, 20, 0, 0, 0,
            5, -5, -10, 0, 0, -10, -5, 5,
            5, 10, 10, -20, -20, 10, 10, 5,
            0, 0, 0, 0, 0, 0, 0, 0
        ]

        # Knight Table (Encourages center posts)
        KNIGHT_PST = [
            -50, -40, -30, -30, -30, -30, -40, -50,
            -40, -20, 0, 0, 0, 0, -20, -40,
            -30, 0, 10, 15, 15, 10, 0, -30,
            -30, 5, 15, 20, 20, 15, 5, -30,
            -30, 0, 15, 20, 20, 15, 0, -30,
            -30, 5, 10, 15, 15, 10, 5, -30,
            -40, -20, 0, 5, 5, 0, -20, -40,
            -50, -40, -30, -30, -30, -30, -40, -50,
        ]

        # Piece values (Centipawns)
        PIECE_VALUES = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }

        # Mapping to PSTs (Simple version: reuse Knight table for Bishop/King center bias)
        PST_MAP = {
            chess.PAWN: PAWN_PST,
            chess.KNIGHT: KNIGHT_PST,
            # Fallback for others to avoid massive code bloat here
            chess.BISHOP: KNIGHT_PST,
            chess.ROOK: [0] * 64,
            chess.QUEEN: [0] * 64,
            chess.KING: [0] * 64
        }

        score = 0

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # 1. Material Value
                val = PIECE_VALUES[piece.piece_type]

                # 2. Positional Value (PST)
                pst_val = 0
                if piece.piece_type in PST_MAP:
                    table = PST_MAP[piece.piece_type]
                    # If White, index is normal. If Black, mirror index (flip vertically)
                    idx = square if piece.color == chess.WHITE else chess.square_mirror(square)
                    pst_val = table[idx]

                if piece.color == chess.WHITE:
                    score += val + pst_val
                else:
                    score -= val + pst_val

        # Normalize to -1.0 to 1.0 (approximate)
        # 1 Pawn advantage (100) becomes ~0.1
        return math.tanh(score / 1000.0)

    def rollout(self, state_tensor) -> float:
        """
        Returns the value estimate from the perspective of the player to move
        at this node.
        """
        if self.board.is_game_over():
            result = self.board.result()
            if result == "1-0":
                return 1.0 if self.board.turn == chess.WHITE else -1.0
            if result == "0-1":
                return -1.0 if self.board.turn == chess.WHITE else 1.0
            return 0.0  # Draw

        logits = self.value_model(state_tensor).detach().cpu().numpy().reshape(-1)

        probs = np.exp(logits) / np.sum(np.exp(logits))

        value = probs[2] - probs[0]

        return value if self.board.turn == chess.WHITE else -value


def MCTS(
        game_state: chess.Board,
        num_sim: int,
        policy_model: DeepForkNet,
        value_model: DeepForkNet,
        device: str,
        seen_states: dict,
        state_history: np.ndarray,
        c_puct: float = 1.5, # The bigger, the more it relies on net prediction
        history_count: int = 1
) -> chess.Move:
    """
    Run a Monte Carlo Tree Search starting from the given game state.

    :param game_state: Starting chess position
    :param num_sim: Number of simulations to run from the root
    :param policy_model: Neural network providing policy and value estimates
    :param device: Torch device identifier (e.g., 'cpu' or 'cuda')
    :param seen_states: Map from state hash to repetition count
    :param state_history: Rolling tensor buffer of prior states
    :param c_puct: Exploration constant for PUCT formula
    :param history_count: Number of historical states to include
    :return: The selected best move from the root after search
    """
    root = MCTSNode(board=game_state, seen_states=seen_states.copy(), state_history=state_history.copy())
    for i in range(num_sim):
        leaf = root

        # Selection
        while leaf.is_expanded:
            best_move = leaf.get_best_move(c_puct)
            leaf = leaf.add_or_get_child(best_move, history_count)

        state_tensor = state_to_tensor(leaf.state_history, leaf.board, leaf.seen_states, history_count)
        state_tensor = torch.from_numpy(state_tensor).float().to(device)
        # Model prediction
        prior_logits = policy_model(state_tensor).detach().to(device).numpy().reshape(-1)

        # Expansion
        if not leaf.is_terminal():
            move_distr = get_move_distribution(prior_logits, leaf.board)
            leaf.expand(move_distr)

        # Rollout
        value_est = value_model(state_tensor).detach().to(device).numpy().reshape(-1)

        # Backpropagation
        leaf.backprop(value_est)
    # visualize_mcts_graph(root)
    for move, (child, est) in root.children.items():
        if child is not None and child.visit_count is not None and child.move is not None and child.total_value is not None:
            print(f"Child {child.move} count: {child.visit_count} value: {child.total_value} est: {child.prior_est}")
        else:
            print(f"move: {move}, est: {est}")
    print("\n\n")
    return root.select_best_child()
