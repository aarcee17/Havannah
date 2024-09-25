import time
import math
import random
import numpy as np
from collections import defaultdict
from typing import Tuple
from helper import get_valid_actions

class Graph:
    def __init__(self, board: np.array):
        self.board = board
        self.dim = board.shape[0]
        self.graph = defaultdict(list)
        self.build_graph()
        
    def get_all_edges(self, dim: int) -> list:

        edges = []

        # Top and bottom edges
        for i in range(dim):
            edges.append((0, i))  # Top edge
            edges.append((dim - 1, i + dim - 1))  # Bottom edge

        # Left and right edges
        for j in range(1, dim - 1):
            edges.append((j, 0))  # Left edge
            edges.append((j + dim - 1, 2 * dim - 2))  # Right edge

        return edges
    
    def get_edge(self, pos: tuple, dim: int) -> int:
        """
        Check if the position is on an edge.
        Returns the edge number if it's on an edge, -1 if it's not.
        """
        i, j = pos
        if i == 0 or i == dim - 1 or j == 0 or j == 2 * dim - 2:
            return 1  # Edge detected
        return -1  # Not on edge

    def build_graph(self):
        for i in range(self.dim):
            for j in range(self.dim):
                if self.board[i, j] in [1, 2]:
                    neighbors = self.get_neighbors(i, j)
                    for ni, nj in neighbors:
                        if self.board[ni, nj] == self.board[i, j]:
                            self.graph[(i, j)].append((ni, nj))

    def get_neighbors(self, i: int, j: int) -> list:
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
        neighbors = []
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.dim and 0 <= nj < self.dim and self.board[ni, nj] != 3:
                neighbors.append((ni, nj))
        return neighbors

    def get_all_corners(self, dim: int) -> list:
        corners = [
            (0, 0), (0, dim - 1),
            (dim - 1, 2 * dim - 2), (2 * dim - 2, dim - 1),
            (dim - 1, 0), (2 * dim - 2, 0)
        ]
        return corners

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

    def get_path_between_nodes(self, start: tuple, end: tuple) -> list:
        visited = set()
        path = []
        def dfs_path(node, target, current_path):
            visited.add(node)
            current_path.append(node)
            if node == target:
                path.extend(current_path)
                return True
            for neighbor in self.graph[node]:
                if neighbor not in visited:
                    if dfs_path(neighbor, target, current_path):
                        return True
            current_path.pop()
            return False
        dfs_path(start, end, [])
        return path


class AIPlayer:
    def __init__(self, player_number: int, timer):
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}: ai'.format(player_number)
        self.timer = timer
        self.max_depth = 3

    def get_move(self, state: np.array) -> Tuple[int, int]:
        best_move = self.mcts(state)
        return best_move

    def mcts(self, state: np.array) -> Tuple[int, int]:
        valid_moves = get_valid_actions(state, self.player_number)
        if not valid_moves:
            raise ValueError("No valid moves available")

        best_move = None
        best_score = -float('inf')
        num_simulations = 100
        for move in valid_moves:
            total_score = 0
            for _ in range(num_simulations):
                new_state = self.simulate_move(state, move, self.player_number)
                score = self.simulate_random_playouts(new_state)
                total_score += score
            average_score = total_score / num_simulations
            if average_score > best_score:
                best_score = average_score
                best_move = move

        return best_move

    def simulate_move(self, state: np.array, move: Tuple[int, int], player: int) -> np.array:
        new_state = state.copy()
        new_state[move[0], move[1]] = player
        return new_state

    def simulate_random_playouts(self, state: np.array) -> float:
        current_player = self.player_number
        num_random_moves = 10
        for _ in range(num_random_moves):
            valid_moves = get_valid_actions(state, current_player)
            if not valid_moves:
                break
            random_move = random.choice(valid_moves)
            state = self.simulate_move(state, random_move, current_player)
            current_player = 2 if current_player == 1 else 1
        return self.evaluate_board(state)

    def evaluate_board(self, state: np.array) -> float:
        ring_score = self.ring_heuristic(state)
        bridge_score = self.bridge_heuristic(state)
        fork_score = self.fork_heuristic(state)
        return ring_score + bridge_score + fork_score

    def ring_heuristic(self, state: np.array) -> int:
        graph = Graph(state)
        components = graph.find_connected_components()
        min_blocks_away = float('inf')
        for component in components:
            if graph.detect_cycles():
                return 0
            else:
                blocks_needed = self.estimate_blocks_for_ring(component, graph, state)
                min_blocks_away = min(min_blocks_away, blocks_needed)
        return min_blocks_away

    def estimate_blocks_for_ring(self, component: list, graph: Graph, state: np.array) -> int:
        empty_neighbors = set()
        for node in component:
            neighbors = graph.get_neighbors(node[0], node[1])
            for neighbor in neighbors:
                if state[neighbor[0], neighbor[1]] == 0:
                    empty_neighbors.add(neighbor)
        return len(empty_neighbors)

    def bridge_heuristic(self, state: np.array) -> int:
        graph = Graph(state)
        corners = graph.get_all_corners(state.shape[0])
        min_blocks_away = float('inf')
        for i in range(len(corners)):
            for j in range(i + 1, len(corners)):
                start_corner = corners[i]
                end_corner = corners[j]
                if graph.get_path_between_nodes(start_corner, end_corner):
                    return 0
                blocks_needed = self.estimate_blocks_for_bridge(start_corner, end_corner, graph, state)
                min_blocks_away = min(min_blocks_away, blocks_needed)
        return min_blocks_away

    def estimate_blocks_for_bridge(self, start: tuple, end: tuple, graph: Graph, state: np.array) -> int:
        path = graph.get_path_between_nodes(start, end)
        if path:
            empty_blocks = [node for node in path if state[node[0], node[1]] == 0]
            return len(empty_blocks)
        return float('inf')

    def fork_heuristic(self, state: np.array) -> int:
        graph = Graph(state)
        edges = graph.get_all_edges(state.shape[0])  # Replace get_edge usage with get_all_edges
        connected_edges = set()

        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                if state[i, j] == self.player_number:
                    if (i, j) in edges:
                        connected_edges.add((i, j))

        if len(connected_edges) >= 3:
            return 0
        return 3 - len(connected_edges)

