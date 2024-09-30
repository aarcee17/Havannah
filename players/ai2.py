import random
import numpy as np
from typing import Tuple
from helper import get_valid_actions, check_win

class AIPlayer:
    
    def __init__(self, player_number: int, timer, max_depth: int = 3):
        """
        Initialize the AIPlayer Agent.
        
        Parameters:
        - player_number (int): Current player number (1 or 2)
        - timer: A Timer object to fetch remaining time
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

        # 3. No immediate win or block, use Minimax with IDS
        return self.minimax_ids(state, valid_moves)

    def minimax_ids(self, state: np.array, valid_moves: list) -> Tuple[int, int]:
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
                score = self.minimax(temp_state, depth, float('-inf'), float('inf'), False)
                if score > best_score:
                    best_score = score
                    best_move = move

        # If no best move is found, fall back to a random valid move
        return best_move or random.choice(valid_moves)

    def minimax(self, state: np.array, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
        """
        Minimax algorithm with alpha-beta pruning.
        
        Parameters:
        - state: np.array
            Current state of the board.
        - depth: int
            Current depth in the search.
        - alpha: float
            Alpha value for alpha-beta pruning.
        - beta: float
            Beta value for alpha-beta pruning.
        - maximizing: bool
            Boolean flag indicating if we are maximizing or minimizing.

        Returns:
        - float: Evaluated score for the state.
        """
        if depth == 0:
            return self.evaluate(state)

        valid_moves = get_valid_actions(state)
        if not valid_moves:
            return self.evaluate(state)

        if maximizing:
            max_eval = float('-inf')
            for move in valid_moves:
                temp_state = np.copy(state)
                temp_state[move[0], move[1]] = self.player_number
                eval = self.minimax(temp_state, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            opponent = 2 if self.player_number == 1 else 1
            for move in valid_moves:
                temp_state = np.copy(state)
                temp_state[move[0], move[1]] = opponent
                eval = self.minimax(temp_state, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def evaluate(self, state: np.array) -> float:
        """
        Evaluation function based on board control, proximity to win, and blocking threats.
        
        Parameters:
        - state: np.array
            Current state of the board.

        Returns:
        - float: Evaluation score of the state.
        """
        score = 0

        # Example evaluation: liberties, control of key positions
        liberties_ai = len(get_valid_actions(state, self.player_number))
        liberties_opponent = len(get_valid_actions(state, 2 if self.player_number == 1 else 1))
        score += (liberties_ai - liberties_opponent) * 0.5  # Weigh board control

        # Further scoring based on proximity to winning conditions (e.g., forks, rings, bridges)

        return score
