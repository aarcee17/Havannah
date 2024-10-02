import random
import numpy as np
from typing import Tuple
from helper import get_valid_actions, check_win

class AIPlayer:
    
    def __init__(self, player_number: int, timer, max_depth: int = 3):
        self.player_number = player_number
        self.type = 'ai2'
        self.player_string = f'Player {player_number}: ai2'
        self.timer = timer
        self.max_depth = max_depth

    def get_move(self, state: np.array) -> Tuple[int, int]:
        valid_moves = get_valid_actions(state)

        # Check if AI can win with this move
        for move in valid_moves:
            temp_state = np.copy(state)
            temp_state[move[0], move[1]] = self.player_number
            if check_win(temp_state, move, self.player_number)[0]:
                print(f"AI selects winning move: {move}")
                return move

        # Check if opponent can win with the next move and block it
        opponent_number = 2 if self.player_number == 1 else 1
        for move in valid_moves:
            temp_state = np.copy(state)
            temp_state[move[0], move[1]] = opponent_number
            if check_win(temp_state, move, opponent_number)[0]:
                print(f"AI blocks opponent's winning move: {move}")
                return move

        # Look for 3-move combinations to block or win
        combo_move = self.lookahead_checkmate(state, valid_moves)
        if combo_move:
            print(f"AI selects strategic move: {combo_move}")
            return combo_move

        # Fallback to MCTS RAVE if no immediate threats or wins are detected
        current_turn = np.count_nonzero(state)
        mcts_move = self.mcts_rave(state, valid_moves, current_turn)
        if mcts_move in valid_moves:
            print(f"AI selects MCTS move: {mcts_move}")
            return mcts_move
        else:
            # As a safety net, return a random valid move
            safe_move = random.choice(valid_moves)
            print(f"AI selects fallback move: {safe_move}")
            return safe_move

    def lookahead_checkmate(self, state: np.array, valid_moves: list) -> Tuple[int, int]:
        opponent_number = 2 if self.player_number == 1 else 1

        for first_move in valid_moves:
            temp_state_1 = np.copy(state)
            temp_state_1[first_move[0], first_move[1]] = self.player_number

            if check_win(temp_state_1, first_move, self.player_number)[0]:
                return first_move

            opponent_moves = get_valid_actions(temp_state_1)
            for second_move in opponent_moves:
                temp_state_2 = np.copy(temp_state_1)
                temp_state_2[second_move[0], second_move[1]] = opponent_number

                if check_win(temp_state_2, second_move, opponent_number)[0]:
                    continue  # Opponent can win; this path is not favorable

                third_moves = get_valid_actions(temp_state_2)
                for third_move in third_moves:
                    temp_state_3 = np.copy(temp_state_2)
                    temp_state_3[third_move[0], third_move[1]] = self.player_number

                    if check_win(temp_state_3, third_move, self.player_number)[0]:
                        return first_move

        return None

    def mcts_rave(self, state: np.array, valid_moves: list, current_turn: int) -> Tuple[int, int]:
        root = self.MCTSNode(state, None, player=self.player_number)
        
        if not root.children:
            self.expand_node(root, self.player_number)

        simulations = 100  # Adjust the number of simulations as needed
        for _ in range(simulations):
            node, state_copy = self.select_node(root)
            reward = self.rollout(state_copy)
            self.backpropagate(node, reward)

        if root.children:
            # Select the move with the highest visit count
            best_child = max(root.children, key=lambda x: x.visits)
            return best_child.move

        # If no children (should not happen), return a random valid move
        return random.choice(valid_moves)

    def select_node(self, node):
        state_copy = np.copy(node.state)
        while node.children:
            node = self.ucb_select(node)
            # Apply the move to the state copy
            state_copy[node.move[0], node.move[1]] = node.player
        return node, state_copy

    def expand_node(self, node, player_number):
        valid_moves = get_valid_actions(node.state)
        opponent_number = 2 if player_number == 1 else 1
        for move in valid_moves:
            temp_state = np.copy(node.state)
            temp_state[move[0], move[1]] = player_number
            child_node = self.MCTSNode(temp_state, move, parent=node, player=opponent_number)
            node.children.append(child_node)

    def ucb_select(self, node):
        total_visits = sum(child.visits for child in node.children)
        ucb_values = [
            (child.value / (child.visits + 1e-5)) + 2 * np.sqrt(np.log(total_visits + 1) / (child.visits + 1e-5))
            for child in node.children
        ]
        max_index = np.argmax(ucb_values)
        return node.children[max_index]

    def rollout(self, state: np.array) -> float:
        # Create a copy of the state to avoid modifying the original
        state_copy = np.copy(state)
        current_player = self.player_number
        opponent_number = 2 if self.player_number == 1 else 1
        for _ in range(10):  # Limit the rollout depth
            valid_moves = get_valid_actions(state_copy)
            if not valid_moves:
                break
            move = random.choice(valid_moves)
            state_copy[move[0], move[1]] = current_player
            if check_win(state_copy, move, current_player)[0]:
                return 1.0 if current_player == self.player_number else 0.0
            # Switch player
            current_player = opponent_number if current_player == self.player_number else self.player_number
        return 0.5  # Return a neutral reward if no winner

    def backpropagate(self, node, reward: float):
        while node is not None:
            node.visits += 1
            if node.player == self.player_number:
                node.value += reward
            else:
                node.value += (1 - reward)
            node = node.parent

    class MCTSNode:
        def __init__(self, state, move, parent=None, player=1):
            self.state = state
            self.move = move
            self.parent = parent
            self.children = []
            self.visits = 0
            self.value = 0.0
            self.player = player  # The player who made the move to reach this state
