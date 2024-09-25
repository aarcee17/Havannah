import time
import math
import random
import numpy as np
from collections import defaultdict
from typing import Tuple
from helper import get_valid_actions, fetch_remaining_time

class Graph:
    def __init__(self, board: np.array, player_number: int):
        self.board = board
        self.dim = board.shape[0]
        self.player_number = player_number
        self.graph = defaultdict(list)
        self.build_graph()
    
    def build_graph(self):
        for i in range(self.dim):
            for j in range(self.dim):
                if self.board[i, j] == self.player_number:
                    neighbors = self.get_neighbors(i, j)
                    for ni, nj in neighbors:
                        if self.board[ni, nj] == self.player_number:
                            self.graph[(i, j)].append((ni, nj))
    
    def get_neighbors(self, i: int, j: int) -> list:
        directions = [(-1, 0), (-1, 1), (0, -1),
                      (0, 1), (1, -1), (1, 0)]
        neighbors = []
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.dim and 0 <= nj < self.dim and self.board[ni, nj] != 3:
                neighbors.append((ni, nj))
        return neighbors
    
    def get_all_corners(self, dim: int) -> list:
        corners = [
            (0, dim - 1),              # Top corner
            (dim - 1, 2 * dim - 2),    # Top-right corner
            (2 * dim - 2, dim - 1),    # Bottom-right corner
            (2 * dim - 2, 0),          # Bottom corner
            (dim - 1, 0),              # Bottom-left corner
            (0, 0)                     # Top-left corner
        ]
        return corners
    
    def get_all_edges(self, dim: int) -> list:
        edges = []
        for i in range(dim):
            edges.append((0, i))  # Top edge
            edges.append((2 * dim - 2, i + dim - 1))  # Bottom edge
        for j in range(1, dim - 1):
            edges.append((j, 0))  # Top-left edge
            edges.append((j + dim - 1, 2 * dim - 2))  # Bottom-right edge
        return edges
    
    def detect_cycles(self) -> bool:
        visited = set()
        rec_stack = set()
        for node in self.graph:
            if node not in visited:
                if self.detect_cycle_dfs(node, visited, rec_stack):
                    return True
        return False
    
    def detect_cycle_dfs(self, node, visited, rec_stack) -> bool:
        visited.add(node)
        rec_stack.add(node)
        for neighbor in self.graph[node]:
            if neighbor not in visited:
                if self.detect_cycle_dfs(neighbor, visited, rec_stack):
                    return True
            elif neighbor in rec_stack:
                return True
        rec_stack.remove(node)
        return False
    
    def find_connected_components(self) -> list:
        visited = set()
        components = []
        for node in self.graph:
            if node not in visited:
                component = self.dfs(node, visited)
                components.append(component)
        return components
    
    def dfs(self, start: tuple, visited=None) -> list:
        if visited is None:
            visited = set()
        stack = [start]
        result = []
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                result.append(node)
                for neighbor in self.graph[node]:
                    if neighbor not in visited:
                        stack.append(neighbor)
        return result
    
    def get_path_between_nodes(self, start: tuple, end: tuple) -> list:
        visited = set()
        path = []
        found = self.dfs_path(start, end, visited, path)
        return path if found else []
    
    def dfs_path(self, node, target, visited, path) -> bool:
        visited.add(node)
        path.append(node)
        if node == target:
            return True
        for neighbor in self.graph[node]:
            if neighbor not in visited:
                if self.dfs_path(neighbor, target, visited, path):
                    return True
        path.pop()
        return False
    
    def hex_distance(self, a: tuple, b: tuple) -> int:
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs((a[0] - a[1]) - (b[0] - b[1])))

class AIPlayer:
    def __init__(self, player_number: int, timer):
        self.player_number = player_number
        self.opponent_number = 2 if player_number == 1 else 1
        self.type = 'ai'
        self.player_string = 'Player {}: ai'.format(player_number)
        self.timer = timer
        self.max_depth = 3
        # Heuristic weights
        self.ring_weight = 1.0
        self.bridge_weight = 0.8
        self.fork_weight = 0.6
        self.expansion_weight = 0.3
        self.opponent_weight = 1.2
        self.num_simulations = 50  # Adjust based on time constraints
    
    def get_move(self, state: np.array) -> Tuple[int, int]:
        best_move = self.mcts(state)
        return best_move
    
    def mcts(self, state: np.array) -> Tuple[int, int]:
        valid_moves = get_valid_actions(state, self.player_number)
        if not valid_moves:
            raise ValueError("No valid moves available")

        best_move = None
        best_score = -float('inf')
        start_time = time.time()
        time_limit = fetch_remaining_time(self.timer, self.player_number) - 1  # Reserve 1 second buffer

        for move in valid_moves:
            total_score = 0
            simulations = 0
            while simulations < self.num_simulations:
                # Check if there's enough time left
                if time.time() - start_time >= time_limit:
                    break
                new_state = self.simulate_move(state, move, self.player_number)
                score = self.simulate_random_playouts(new_state)
                total_score += score
                simulations += 1
            if simulations == 0:
                average_score = -float('inf')
            else:
                average_score = total_score / simulations
            if average_score > best_score:
                best_score = average_score
                best_move = move

        if best_move is None:
            best_move = random.choice(valid_moves)
        return best_move
    
    def simulate_move(self, state: np.array, move: Tuple[int, int], player: int) -> np.array:
        new_state = state.copy()
        new_state[move[0], move[1]] = player
        return new_state
    
    def simulate_random_playouts(self, state: np.array) -> float:
        current_player = self.player_number
        num_random_moves = 5  # Keep small due to time constraints
        for _ in range(num_random_moves):
            valid_moves_p1 = get_valid_actions(state, self.player_number)
            valid_moves_p2 = get_valid_actions(state, self.opponent_number)
            if not valid_moves_p1 or not valid_moves_p2:
                break
            move_p1 = random.choice(valid_moves_p1)
            state = self.simulate_move(state, move_p1, self.player_number)
            move_p2 = random.choice(valid_moves_p2)
            state = self.simulate_move(state, move_p2, self.opponent_number)
        return self.evaluate_board(state)
    
    def evaluate_board(self, state: np.array) -> float:
        # AI's scores
        ring_score = self.ring_heuristic(state)
        bridge_score = self.bridge_heuristic(state)
        fork_score = self.fork_heuristic(state)
        expansion_score = self.expansion_heuristic(state)
        ai_score = (self.ring_weight * ring_score +
                    self.bridge_weight * bridge_score +
                    self.fork_weight * fork_score -
                    self.expansion_weight * expansion_score)
        # Opponent's scores
        opponent_state = self.get_opponent_state(state)
        opp_ring_score = self.ring_heuristic(opponent_state)
        opp_bridge_score = self.bridge_heuristic(opponent_state)
        opp_fork_score = self.fork_heuristic(opponent_state)
        opp_expansion_score = self.expansion_heuristic(opponent_state)
        opponent_score = (self.ring_weight * opp_ring_score +
                          self.bridge_weight * opp_bridge_score +
                          self.fork_weight * opp_fork_score -
                          self.expansion_weight * opp_expansion_score)
        # Combine scores (lower is better for AI)
        total_score = opponent_score * self.opponent_weight - ai_score
        return total_score
    
    def get_opponent_state(self, state: np.array) -> np.array:
        opponent_state = state.copy()
        opponent_state[opponent_state == self.player_number] = self.opponent_number
        opponent_state[opponent_state == self.opponent_number] = self.player_number
        return opponent_state
    
    # Improved Ring Heuristic
    def ring_heuristic(self, state: np.array) -> float:
        graph = Graph(state, self.player_number)
        if graph.detect_cycles():
            return 0  # Ring already formed
        # Estimate moves to form a ring
        # For simplicity, return a high value indicating it's hard
        return 10.0  # Placeholder value; adjust based on better estimation
    
    # Improved Bridge Heuristic
    def bridge_heuristic(self, state: np.array) -> float:
        graph = Graph(state, self.player_number)
        corners = graph.get_all_corners(state.shape[0])
        min_moves_to_bridge = float('inf')
        for i in range(len(corners)):
            for j in range(i + 1, len(corners)):
                start_corner = corners[i]
                end_corner = corners[j]
                moves_needed = self.moves_to_form_bridge(start_corner, end_corner, state)
                min_moves_to_bridge = min(min_moves_to_bridge, moves_needed)
        return min_moves_to_bridge
    
    def moves_to_form_bridge(self, start: tuple, end: tuple, state: np.array) -> int:
        from queue import PriorityQueue
        open_set = PriorityQueue()
        open_set.put((0, start))
        came_from = {}
        g_score = {start: 0}
        while not open_set.empty():
            _, current = open_set.get()
            if current == end:
                return g_score[current]  # Path found
            neighbors = self.get_neighbors(current[0], current[1], state)
            for neighbor in neighbors:
                if state[neighbor[0], neighbor[1]] in [self.player_number, 0]:
                    tentative_g_score = g_score[current] + 1
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + self.heuristic(neighbor, end)
                        open_set.put((f_score, neighbor))
        return 10  # Path not found; return high value
    
    def heuristic(self, a: tuple, b: tuple) -> int:
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))
    
    def get_neighbors(self, i: int, j: int, state: np.array) -> list:
        directions = [(-1, 0), (-1, 1), (0, -1),
                      (0, 1), (1, -1), (1, 0)]
        neighbors = []
        dim = state.shape[0]
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < dim and 0 <= nj < dim and state[ni, nj] != 3:
                neighbors.append((ni, nj))
        return neighbors
    
    # Improved Fork Heuristic
    def fork_heuristic(self, state: np.array) -> float:
        graph = Graph(state, self.player_number)
        all_edges = graph.get_all_edges(state.shape[0])
        connected_edges = self.get_connected_edges(graph, state)
        if len(connected_edges) >= 3:
            return 0  # Fork already formed
        moves_needed = self.moves_to_connect_edges(graph, state, connected_edges, all_edges)
        return moves_needed
    
    def get_connected_edges(self, graph: Graph, state: np.array) -> set:
        connected_edges = set()
        edges = graph.get_all_edges(state.shape[0])
        for pos in graph.graph.keys():
            if pos in edges:
                connected_edges.add(pos)
        return connected_edges
    
    def moves_to_connect_edges(self, graph: Graph, state: np.array, connected_edges: set, all_edges: list) -> int:
        unconnected_edges = set(all_edges) - connected_edges
        component_positions = [pos for pos in graph.graph.keys()]
        edge_distances = {}
        for edge in unconnected_edges:
            min_distance = float('inf')
            for pos in component_positions:
                distance = self.hex_distance(pos, edge)
                if distance < min_distance:
                    min_distance = distance
            edge_distances[edge] = min_distance
        distances = sorted(edge_distances.values())
        moves_needed = sum(distances[:3 - len(connected_edges)])  # Moves to connect to enough edges
        return moves_needed
    
    def hex_distance(self, a: tuple, b: tuple) -> int:
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs((a[0] + a[1]) - (b[0] + b[1])))
    
    # Expansion Potential Heuristic
    def expansion_heuristic(self, state: np.array) -> float:
        graph = Graph(state, self.player_number)
        components = graph.find_connected_components()
        total_potential = 0
        for component in components:
            potential = self.expansion_potential(component, state)
            total_potential += potential
        return total_potential
    
    def expansion_potential(self, component: list, state: np.array) -> int:
        max_steps = 2
        visited = set(component)
        queue = [(node, 0) for node in component]
        potential = 0
        while queue:
            node, steps = queue.pop(0)
            if steps >= max_steps:
                continue
            neighbors = self.get_neighbors(node[0], node[1], state)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    if state[neighbor[0], neighbor[1]] == 0:
                        potential += 1
                        queue.append((neighbor, steps + 1))
        return potential

