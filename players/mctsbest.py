import random
import numpy as np
from typing import Tuple
from helper import get_valid_actions, check_win, get_neighbours, get_all_edges, get_all_corners, get_edge, get_corner

class AIPlayer:
    
    def __init__(self, player_number: int, timer, max_depth: int = 3, heuristic_weight=0.5):
        self.player_number = player_number
        self.type = 'ai2'
        self.player_string = f'Player {player_number}: ai2'
        self.timer = timer
        self.max_depth = max_depth
        self.heuristic_weight = heuristic_weight  # Weight for heuristic bias in UCB

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

        # Fallback to MCTS RAVE with heuristics if no immediate threats or wins are detected
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
            if node.move is not None:
                state_copy[node.move[0], node.move[1]] = node.player
        return node, state_copy

    def expand_node(self, node, player_number):
        valid_moves = get_valid_actions(node.state)
        opponent_number = 2 if player_number == 1 else 1
        for move in valid_moves:
            temp_state = np.copy(node.state)
            temp_state[move[0], move[1]] = player_number
            heuristic_value = self.evaluate_move_heuristic(temp_state, move, player_number)
            child_node = self.MCTSNode(temp_state, move, parent=node, player=opponent_number, heuristic_value=heuristic_value)
            node.children.append(child_node)

    def ucb_select(self, node):
        total_visits = sum(child.visits for child in node.children) + 1
        ucb_values = []
        for child in node.children:
            exploitation = (child.value / (child.visits + 1e-5))
            exploration = 2 * np.sqrt(np.log(total_visits) / (child.visits + 1e-5))
            heuristic_bias = (child.heuristic_value * self.heuristic_weight) / (child.visits + 1)
            ucb_value = exploitation + exploration + heuristic_bias
            ucb_values.append(ucb_value)
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

            # Heuristic rollout policy
            moves_with_heuristics = []
            for move in valid_moves:
                heuristic_value = self.evaluate_move_heuristic(state_copy, move, current_player)
                moves_with_heuristics.append((move, heuristic_value))

            # Select move with highest heuristic value
            if moves_with_heuristics:
                moves_with_heuristics.sort(key=lambda x: x[1], reverse=True)
                # Optionally, select among top few moves randomly
                top_moves = [m for m in moves_with_heuristics if m[1] == moves_with_heuristics[0][1]]
                move = random.choice(top_moves)[0]
            else:
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

    def evaluate_move_heuristic(self, state, move, player_number):
        heuristic_value = 0

        # Heuristic 1: Locality (playing near own stones)
        own_stones = np.argwhere(state == player_number)
        if own_stones.size > 0:
            distances = np.abs(own_stones - move).sum(axis=1)
            min_distance = np.min(distances)
            if min_distance == 1:
                heuristic_value += 3  # Direct neighbor
            elif min_distance == 2:
                heuristic_value += 2  # Virtual connection
            elif min_distance == 3:
                heuristic_value += 1  # Distance 2 but not VC

        # Heuristic 2: Edge Connectivity
        # Check if move is on an edge or corner
        dim = (state.shape[0] + 1) // 2
        if self.is_edge(move, dim):
            heuristic_value += 2  # Bonus for being on an edge
        if self.is_corner(move, dim):
            heuristic_value += 3  # Bonus for being on a corner

        # Heuristic 3: Group Size (approximate)
        # Bonus for connecting to own stones
        neighbors = self.get_neighbors(move, dim)
        own_neighbors = [n for n in neighbors if state[n[0], n[1]] == player_number]
        heuristic_value += len(own_neighbors)  # Bonus for each own neighbor

        return heuristic_value

    def is_edge(self, pos, dim):
        # Return True if position is on an edge (but not a corner)
        edge = get_edge(pos, dim)
        if edge != -1 and not self.is_corner(pos, dim):
            return True
        else:
            return False

    def is_corner(self, pos, dim):
        corner = get_corner(pos, dim)
        return corner != -1

    def get_neighbors(self, pos, dim):
        return get_neighbours(dim, pos)

    class MCTSNode:
        def __init__(self, state, move, parent=None, player=1, heuristic_value=0):
            self.state = state
            self.move = move
            self.parent = parent
            self.children = []
            self.visits = 0
            self.value = 0.0
            self.player = player  # The player who made the move to reach this state
            self.heuristic_value = heuristic_value  # The heuristic value for this node
