import random
import numpy as np
from typing import Tuple
from helper import get_valid_actions, check_win

class AIPlayer:
    
    def __init__(self, player_number: int, timer):
        """
        Initialize the AIPlayer Agent.
        
        Parameters:
        - player_number (int): Current player number (1 or 2)
        - timer: A Timer object to fetch remaining time
        """
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = f'Player {player_number}: ai'
        self.timer = timer

    def get_move(self, state: np.array) -> Tuple[int, int]:
        """
        Given the current state of the board, return the next move.
        
        Parameters:
        - state: np.array
            A numpy array representing the board state.

        Returns:
        - Tuple[int, int]: Coordinates of the board cell where the move will be placed.
        """
        # Get all valid moves from the current board state
        valid_moves = get_valid_actions(state)
        
        # Check if AI has a winning move
        for move in valid_moves:
            temp_state = np.copy(state)
            temp_state[move[0], move[1]] = self.player_number

            # Check if this move results in a win
            if check_win(temp_state, move, self.player_number)[0]:
                print(f"AI winning move: {move}")
                return move

        # Check if opponent is one move away from winning and block it
        opponent_number = 2 if self.player_number == 1 else 1
        for move in valid_moves:
            temp_state = np.copy(state)
            temp_state[move[0], move[1]] = opponent_number

            # Check if this move results in a win for the opponent
            if check_win(temp_state, move, opponent_number)[0]:
                print(f"Blocking opponent's move: {move}")
                return move
        
        # If no immediate win or block is found, choose a random valid move
        if valid_moves:
            random_move = random.choice(valid_moves)
            print(f"Random move: {random_move}")
            return random_move

        # No valid moves, return None
        return None

