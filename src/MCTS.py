"""
Monte Carlo Tree Search (MCTS) implementation for the DeepFork chess agent.

This module defines the MCTSNode structure and the MCTS search function that
leverages policy and value estimates from the DeepForkNet model to guide search.
"""

from __future__ import annotations
from typing import Hashable
from utils.chess_utils import state_to_tensor, update_history, get_move_distribution
from model import DeepForkNet
from utils.model_utils import visualize_mcts_graph

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


    def backprop(self, value: np.float32):
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


def MCTS(
        game_state: chess.Board,
        num_sim: int,
        model: DeepForkNet,
        device: str,
        seen_states: dict,
        state_history: np.ndarray,
        history_count: int,
        c_puct: float = 1, # The bigger, the more it relies on net prediction
) -> chess.Move:
    """
    Run a Monte Carlo Tree Search starting from the given game state.

    :param game_state: Starting chess position
    :param num_sim: Number of simulations to run from the root
    :param model: Neural network providing policy and value estimates
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
        # Move from root to best leaf
        while leaf.is_expanded:
            best_move = leaf.get_best_move(c_puct)
            leaf = leaf.add_or_get_child(best_move, history_count)

        state_tensor = state_to_tensor(leaf.state_history, leaf.board, leaf.seen_states, history_count)
        state_tensor = torch.from_numpy(state_tensor).float().to(device)
        # Model prediction
        value_est, prior_ests = model(state_tensor)
        if root == leaf:
            print(f"Value estimation: {value_est}")

        # Convert torch tensors into numpy shapes
        prior_ests = prior_ests.detach().to(device).numpy().reshape(-1)
        value_est = value_est.item()

        if not leaf.is_terminal():
            move_distr = get_move_distribution(prior_ests, leaf.board)
            leaf.expand(move_distr)

        # Backpropagation
        leaf.backprop(value_est)
    visualize_mcts_graph(root)
    for move, (child, est) in root.children.items():
        if child is not None and child.visit_count is not None and child.move is not None and child.total_value is not None:
            print(f"Child {child.move} count: {child.visit_count} value: {child.total_value} est: {child.prior_est}")
        else:
            print(f"move: {move}, est: {est}")
    print("\n\n")
    return root.select_best_child()
