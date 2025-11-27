from __future__ import annotations
from typing import Hashable
from utils.chess_utils import state_to_tensor, update_history, get_move_distribution
from model import DeepForkNet

import math
import torch
import chess
import numpy as np


class MCTSNode:
    def __init__(
            self,
            board: chess.Board,
            seen_states: dict[Hashable, int],
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
        self.num_children = 0
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
            puct = self.Q() + self.U(c_puct, est, child_visit_count)
            if best_value < puct:
                best_move = move
                best_value = puct
        return best_move


    def add_or_get_child(self, move: chess.Move, history_count: int) -> MCTSNode:
        child, est = self.children[move]
        if child is None:
            self.num_children += 1
            if self.num_children == len(self.children):
                self.is_expanded = True
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


    def expand(self, move_distr: dict[chess.Move, np.float32]) -> None:
        """
        Expand the current node by creating all of its children objects.
        :param move_distr: Policy head estimations from the model
        """

        self.children = {move: [None, est] for move, est in move_distr.items()}


    def add_dirichlet_noise(self, prior_ests: dict[int, np.float32]):
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

def MCTS(
        game_state: chess.Board,
        num_sim: int,
        model: DeepForkNet,
        seen_states: dict[Hashable, int],
        state_history: np.ndarray,
        c_puct: float = 1.0,
        history_count: int = 8
) -> chess.Move:
    root = MCTSNode(board=game_state, seen_states=seen_states.copy(), state_history=state_history.copy())
    for i in range(num_sim):
        leaf = root
        # Move from root to best leaf
        while leaf.is_expanded:
            best_move = leaf.get_best_move(c_puct)
            leaf = leaf.add_or_get_child(best_move, history_count)

        state_tensor = state_to_tensor(leaf.state_history, leaf.board, leaf.seen_states, history_count)
        state_tensor = torch.from_numpy(state_tensor).float().cpu()
        # Model prediction
        prior_ests, value_est = model(state_tensor)

        # Convert torch tensors into numpy shapes
        prior_ests = prior_ests.detach().cpu().numpy().reshape(-1)
        value_est = value_est.item()

        if not leaf.is_terminal():
            move_distr = get_move_distribution(prior_ests, leaf.board)
            leaf.expand(move_distr)

        # Backpropagation
        leaf.backprop(value_est)
    return max(root.children, key=lambda child: child.visit_count)
