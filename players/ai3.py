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
        self.type = 'ai2'
        self.player_string = f'Player {player_number}: ai2'
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


import random
import numpy as np
from typing import Tuple
from helper import get_valid_actions

class AIPlayer:
    def __init__(self, player_number: int, timer, max_depth=4):
        self.player_number = player_number
        self.opponent_number = 2 if player_number == 1 else 1
        self.type = 'ai'
        self.player_string = 'Player {}: ai'.format(player_number)
        self.timer = timer
        self.max_depth = max_depth
        self.move_count = 0
        
        self.opening_moves_player_1_a2 = [(2, 0), (1, 0), (1, 1)]  # a2, a1, b2
        self.opening_moves_player_2_a2 = [(1, 4), (3, 2), (4, 1)]  # b5, d3, e2

    def get_move(self, state: np.array) -> Tuple[int, int]:
        if state.shape[0] == 5 and self.move_count < 1:
            return self.get_opening_move(state)
        mate_in_one_move = self.check_mate_in_one(state)
        if mate_in_one_move is not None:
            return mate_in_one_move            
        return self.iterative_deepening_search(state)

    def check_mate_in_one(self, state: np.array) -> Tuple[int, int]:
        # Check if we can win with a mate-in-one
        for move in get_valid_actions(state, self.player_number):
            new_state = self.simulate_move(state, move, self.player_number)
            if check_win(new_state, self.player_number):
                return move

        # Check if the opponent can win with a mate-in-one and block it
        for move in get_valid_actions(state, self.opponent_number):
            new_state = self.simulate_move(state, move, self.opponent_number)
            if check_win(new_state, self.opponent_number):
                return move
                
    def get_opening_move(self, state: np.array) -> Tuple[int, int]:
        if self.player_number == 1:
            move = self.opening_moves_player_1_a2[self.move_count]
        else:
            move = self.opening_moves_player_2_a2[self.move_count]
        self.move_count += 1
        return move

    def iterative_deepening_search(self, state: np.array) -> Tuple[int, int]:
        best_move = None
        best_score = -float('inf')
        for depth in range(1, self.max_depth + 1):
            move, score = self.minimax(state, depth, True, -float('inf'), float('inf'))
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def minimax(self, state: np.array, depth: int, is_maximizing: bool, alpha: float, beta: float) -> Tuple[Tuple[int, int], float]:
        valid_moves = get_valid_actions(state, self.player_number if is_maximizing else self.opponent_number)
        if depth == 0 or not valid_moves:
            return None, self.evaluate_board(state)

        best_move = None
        if is_maximizing:
            best_score = -float('inf')
            for move in valid_moves:
                new_state = self.simulate_move(state, move, self.player_number)
                _, score = self.minimax(new_state, depth - 1, False, alpha, beta)
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
            return best_move, best_score
        else:
            best_score = float('inf')
            for move in valid_moves:
                new_state = self.simulate_move(state, move, self.opponent_number)
                _, score = self.minimax(new_state, depth - 1, True, alpha, beta)
                if score < best_score:
                    best_score = score
                    best_move = move
                beta = min(beta, score)
                if beta <= alpha:
                    break
            return best_move, best_score

    def simulate_move(self, state: np.array, move: Tuple[int, int], player: int) -> np.array:
        new_state = state.copy()
        new_state[move[0], move[1]] = player
        return new_state

    def evaluate_board(self, state: np.array) -> float:
        """
        Evaluate the board state by summing heuristic values based on:
        - Virtual Connections (value 100)
        - Connectivity (value 20)
        - Locality (value 3)
        - Local Reply (value 5)
        - Distance (value 2)
        - Group Size (value 2)
        """
        virtual_connections_score = self.virtual_connections(state, self.player_number)
        connectivity_score = self.connectivity(state, self.player_number)
        locality_score = self.locality(state, self.player_number)
        local_reply_score = self.local_reply(state, self.player_number)
        distance_score = self.distance_heuristic(state, self.player_number)
        group_size_score = self.group_size(state, self.player_number)
        
        # Combine all heuristic scores
        total_score = (10 * virtual_connections_score) + \
                      (5 * connectivity_score) + \
                      (3 * locality_score) + \
                      (5 * local_reply_score) + \
                      (2 * distance_score) + \
                      (2 * group_size_score)
        
        # Opponent's score (subtract their potential)
        opponent_total_score = self.evaluate_opponent(state)
        
        # Maximize our potential and minimize the opponent's potential
        return total_score - opponent_total_score

    def evaluate_opponent(self, state: np.array) -> float:
        """
        Evaluates the opponent's board by using the same heuristics as for the player.
        """
        opponent_virtual_connections = self.virtual_connections(state, self.opponent_number)
        opponent_connectivity = self.connectivity(state, self.opponent_number)
        opponent_locality = self.locality(state, self.opponent_number)
        opponent_local_reply = self.local_reply(state, self.opponent_number)
        opponent_distance = self.distance_heuristic(state, self.opponent_number)
        opponent_group_size = self.group_size(state, self.opponent_number)
        
        return (10 * opponent_virtual_connections) + \
               (5 * opponent_connectivity) + \
               (3 * opponent_locality) + \
               (5 * opponent_local_reply) + \
               (2 * opponent_distance) + \
               (2 * opponent_group_size)

    # Fixed version of virtual_connections and helper methods
    def virtual_connections(self, state: np.array, player_number: int) -> float:
        groups = self.get_connected_groups(state, player_number)
        total_virtual_score = 0
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                group1 = groups[i]
                group2 = groups[j]
                moves_needed = self.estimate_virtual_connection(group1, group2, state)
                if moves_needed == 0:
                    total_virtual_score += 10
                elif moves_needed == 1:
                    total_virtual_score += 8
                elif moves_needed == 2:
                    total_virtual_score += 5
                else:
                    total_virtual_score += 1
        return total_virtual_score

    def estimate_virtual_connection(self, group1: list, group2: list, state: np.array) -> int:
        from collections import deque
        queue = deque([(pos, 0) for pos in group1])
        visited = set(group1)
        while queue:
            current, steps = queue.popleft()
            if current in group2:
                return steps
            neighbors = self.get_neighbors(current[0], current[1], state)
            for neighbor in neighbors:
                if neighbor not in visited and state[neighbor[0], neighbor[1]] == 0:
                    visited.add(neighbor)
                    queue.append((neighbor, steps + 1))
        return float('inf')

    # Helper function: Connectivity, Locality, etc.
    # (Implementations for other heuristics)
    def connectivity(self, state: np.array, player_number: int) -> float:
        """
        Evaluates the connectivity of the player's stones.
        Higher connectivity means that the player's stones are well-connected.
        """
        groups = self.get_connected_groups(state, player_number)
        total_connectivity_score = 0

        for group in groups:
            group_size = len(group)
            if group_size >= 6:
                total_connectivity_score += 20  # Large connected group
            elif group_size >= 4:
                total_connectivity_score += 10  # Medium connected group
            else:
                total_connectivity_score += 2   # Small connected group

        return total_connectivity_score

    def locality(self, state: np.array, player_number: int) -> float:
        """
        Favors moves that are close to the player's existing pieces.
        This helps reinforce existing structures or making progress towards a winning condition.
        """
        groups = self.get_connected_groups(state, player_number)
        locality_score = 0

        for group in groups:
            for stone in group:
                neighbors = self.get_neighbors(stone[0], stone[1], state)
                empty_neighbors = sum(1 for neighbor in neighbors if state[neighbor[0], neighbor[1]] == 0)
                locality_score += empty_neighbors

        return locality_score

    def local_reply(self, state: np.array, player_number: int) -> float:
        """
        This heuristic measures whether the player is responding to the opponent's recent moves.
        Encourages blocking or countering the opponent's progress by placing stones near them.
        """
        opponent_moves = self.get_opponent_recent_moves(state)
        local_reply_score = 0

        for opponent_move in opponent_moves:
            neighbors = self.get_neighbors(opponent_move[0], opponent_move[1], state)
            local_reply_score += sum(1 for neighbor in neighbors if state[neighbor[0], neighbor[1]] == player_number)

        return local_reply_score

    def get_opponent_recent_moves(self, state: np.array) -> list:
        """
        Finds recent moves made by the opponent by checking the board's current state.
        """
        opponent_moves = []
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                if state[i, j] == self.opponent_number:
                    opponent_moves.append((i, j))
        return opponent_moves

    def distance_heuristic(self, state: np.array, player_number: int) -> float:
        """
        This heuristic minimizes the distance between the player's stones.
        The closer the player's stones are to each other, the more control they have over the board.
        """
        groups = self.get_connected_groups(state, player_number)
        total_distance_score = 0

        for group in groups:
            if len(group) > 1:
                total_distance = 0
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        total_distance += abs(group[i][0] - group[j][0]) + abs(group[i][1] - group[j][1])
                avg_distance = total_distance / (len(group) * (len(group) - 1) / 2)
                total_distance_score += (1 / (avg_distance + 1))  # Inverse of average distance

        return total_distance_score

    def group_size(self, state: np.array, player_number: int) -> float:
        """
        Favors larger groups of connected stones. Larger groups offer more board control and are closer to forming winning structures.
        """
        groups = self.get_connected_groups(state, player_number)
        total_group_size_score = 0

        for group in groups:
            total_group_size_score += len(group) ** 2  # Exponential reward for large groups

        return total_group_size_score

    def get_neighbors(self, i: int, j: int, state: np.array) -> list:
        """
        Returns a list of neighboring cells for the given position (i, j).
        Only valid (within the board) and non-blocked neighbors are returned.
        """
        directions = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]
        neighbors = []
        dim = state.shape[0]
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < dim and 0 <= nj < dim and state[ni, nj] != 3:
                neighbors.append((ni, nj))
        return neighbors

    def get_connected_groups(self, state: np.array, player_number: int) -> list:
        """
        Returns a list of connected groups of stones for the given player.
        Each group is a list of tuples representing the positions of the stones in that group.
        """
        visited = set()
        groups = []
        dim = state.shape[0]

        for i in range(dim):
            for j in range(dim):
                if (i, j) not in visited and state[i, j] == player_number:
                    group = self.dfs(state, i, j, player_number, visited)
                    groups.append(group)

        return groups

    def dfs(self, state: np.array, i: int, j: int, player_number: int, visited: set) -> list:
        """
        Performs a depth-first search (DFS) to find all connected stones of the player.
        Returns the group of connected stones as a list of positions.
        """
        stack = [(i, j)]
        group = []

        while stack:
            x, y = stack.pop()
            if (x, y) not in visited:
                visited.add((x, y))
                group.append((x, y))
                neighbors = self.get_neighbors(x, y, state)
                for neighbor in neighbors:
                    if state[neighbor[0], neighbor[1]] == player_number:
                        stack.append(neighbor)

        return group
