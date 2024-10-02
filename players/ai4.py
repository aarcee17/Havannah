           

#########

import random
import numpy as np
from typing import Tuple
from helper import get_valid_actions, check_win, check_ring, get_edge
from helper import *

class AIPlayer:
    
    def __init__(self, player_number: int, timer, max_depth: int = 3):
        """
        Initialize the AIPlayer Agent.
        
        Parameters:
        - player_number (int): Current player number (1 or 2)
        - timer: A Timer object to fetch remaining time (currently unused)
        - max_depth (int): Maximum depth for Minimax with IDS
        """
        self.player_number = player_number
        self.type = 'ai2'
        self.player_string = f'Player {player_number}: ai2'
        self.timer = timer
        self.max_depth = max_depth

    def get_move(self, state: np.array) -> Tuple[int, int]:
        """
        Given the current state of the board, return the next move.
        
        Parameters:
        - state: np.array
            A numpy array representing the board state.

        Returns:
        - Tuple[int, int]: Coordinates of the board cell where the move will be placed.
        """
        valid_moves = get_valid_actions(state)

        # 1. Check if AI can win with this move
        for move in valid_moves:
            temp_state = np.copy(state)
            temp_state[move[0], move[1]] = self.player_number

            if check_win(temp_state, move, self.player_number)[0]:
                print(f"AI winning move: {move}")
                return move

        # 2. Check if opponent can win with the next move and block it
        opponent_number = 2 if self.player_number == 1 else 1
        for move in valid_moves:
            temp_state = np.copy(state)
            temp_state[move[0], move[1]] = opponent_number

            if check_win(temp_state, move, opponent_number)[0]:
                print(f"Blocking opponent's move: {move}")
                return move
        
        current_turn = np.count_nonzero(state)

        # 3. No immediate win or block, use Minimax with IDS
        return self.minimax_ids(state, valid_moves, current_turn)

    def minimax_ids(self, state: np.array, valid_moves: list, current_turn: int) -> Tuple[int, int]:
        """
        Use Minimax with Iterative Deepening Search to find the best move.
        
        Parameters:
        - state: np.array
            Current state of the board.
        - valid_moves: list of valid moves.

        Returns:
        - Tuple[int, int]: Best move based on Minimax with IDS.
        """
        best_move = None
        best_score = float('-inf')

        # Perform Iterative Deepening Search
        for depth in range(1, self.max_depth + 1):
            for move in valid_moves:
                temp_state = np.copy(state)
                temp_state[move[0], move[1]] = self.player_number
                score = self.minimax(temp_state, depth, float('-inf'), float('inf'), False, current_turn)

                if score > best_score:
                    best_score = score
                    best_move = move

        # If no best move is found, fall back to a random valid move
        return best_move or random.choice(valid_moves)

    def minimax(self, state: np.array, depth: int, alpha: float, beta: float, maximizing_player: bool, current_turn: int) -> float:
        if depth == 0:  # Stop the recursion when the depth limit is reached
            return self.evaluate(state, current_turn)
        
        valid_moves = get_valid_actions(state)
        
        if maximizing_player:
            max_eval = float('-inf')
            for move in valid_moves:
                temp_state = np.copy(state)
                temp_state[move[0], move[1]] = self.player_number
                eval = self.minimax(temp_state, depth - 1, alpha, beta, False, current_turn + 1)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            opponent_number = 2 if self.player_number == 1 else 1
            for move in valid_moves:
                temp_state = np.copy(state)
                temp_state[move[0], move[1]] = opponent_number
                eval = self.minimax(temp_state, depth - 1, alpha, beta, True, current_turn + 1)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def evaluate(self, state: np.array, current_turn: int) -> float:
        """
        Evaluation function based on proximity to winning conditions (bridges, rings), diamond head formation,
        and multiple rollouts.
        
        Parameters:
        - state: np.array
            Current state of the board.
        - current_turn: int
            Current turn number in the game to account for ring rule depth.
        
        Returns:
        - float: Evaluation score of the state.
        """
        score = 0
        dim = state.shape[0]
        player_num = self.player_number
        
        # Edge connection bonus (to encourage bridge formation)
        for move in get_valid_actions(state, player_num):
            if get_edge(move, dim) != -1:  # Moves close to the edges
                score += 5  # Encourage edge connectivity for bridge potential
        
        # Ring formation (only score after 70% of the board is filled)
        total_cells = dim * dim
        filled_cells = np.count_nonzero(state)
        ring_rule_threshold = 0.7 * total_cells

        if filled_cells >= ring_rule_threshold:
            for move in get_valid_actions(state, player_num):
                temp_state = np.copy(state)
                temp_state[move[0], move[1]] = player_num
                if check_ring(temp_state, move):  # If forming a ring
                    score += 20  # High reward for forming or nearing a ring

        # Encourage moves that contribute to forming permanent stones in a ring (3 permanent stones)
        if filled_cells >= ring_rule_threshold:
            for move in get_valid_actions(state, player_num):
                temp_state = np.copy(state)
                temp_state[move[0], move[1]] = player_num
                if check_ring(temp_state, move):
                    score += 10  # Extra bonus for permanent stones
        
        # Diamond head strategy
        for move in get_valid_actions(state, player_num):
            if self.forms_diamond_head(state, move, player_num):
                score += 5  # Bonus for forming diamond heads (strategically strong positions)

        return score

    def are_connected(self, state: np.array, cell1: Tuple[int, int], cell2: Tuple[int, int]) -> bool:
        """
        Check if two cells are connected on the board for a specific player using BFS.
        
        Parameters:
        - state: np.array
            Current state of the board.
        - cell1: Tuple[int, int]
            Coordinates of the first cell.
        - cell2: Tuple[int, int]
            Coordinates of the second cell.
        
        Returns:
        - bool: True if the two cells are connected, False otherwise.
        """
        dim = state.shape[0]
        player_num = state[cell1[0], cell1[1]]

        # Return False if either of the cells is not occupied by the same player
        if state[cell2[0], cell2[1]] != player_num:
            return False

        # Use BFS to check connection between cell1 and cell2
        visited = set()
        queue = [cell1]

        while queue:
            current = queue.pop(0)
            
            # If we reach cell2, they are connected
            if current == cell2:
                return True
            
            visited.add(current)

            # Check all neighboring cells
            for neighbor in get_neighbours(dim, current):
                if neighbor not in visited and state[neighbor[0], neighbor[1]] == player_num:
                    queue.append(neighbor)

        # If BFS completes without finding a path, they are not connected
        return False

    def forms_diamond_head(self, state: np.array, move: Tuple[int, int], player_num: int) -> bool:
        """
        Check if a move creates a diamond head structure.
        
        Parameters:
        - state: np.array
            Current state of the board.
        - move: Tuple[int, int]
            Coordinates of the potential move.
        - player_num: int
            Player number (1 or 2).
        
        Returns:
        - bool: True if the move forms a diamond head, False otherwise.
        """
        dim = state.shape[0]
        neighbors = get_neighbours(dim, move)
        
        # Check if two opposite neighbors are unconnected but can be connected via this move
        for i, neighbor1 in enumerate(neighbors):
            for neighbor2 in neighbors[i+1:]:
                if (state[neighbor1[0], neighbor1[1]] == player_num and
                    state[neighbor2[0], neighbor2[1]] == player_num and
                    not self.are_connected(state, neighbor1, neighbor2)):
                    return True

        return False
