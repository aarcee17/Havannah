import random
import numpy as np
from typing import Tuple
from helper import (
    get_valid_actions,
    check_win,
    get_neighbours,
    get_all_edges,
    get_all_corners,
    get_edge,
    get_corner,
    fetch_remaining_time,
)
import time
from collections import deque

class AIPlayer:
    def __init__(self, player_number: int, timer, max_depth: int = 3, heuristic_weight=0.5):
        self.player_number = player_number
        self.type = 'ai2'
        self.player_string = f'Player {player_number}: ai2'
        self.timer = timer
        self.max_depth = max_depth
        self.heuristic_weight = heuristic_weight  # Weight for heuristic bias in UCB
        self.root = None  # Root of the MCTS tree
        self.num_rollouts_per_simulation = 5  # Number of rollouts per simulation
        self.time_per_move = 5  # Time allocated per move in seconds
        self.lgrf_table = {}  # Last Good Reply with Forgetting
        self.opponent_last_move = None  # Track opponent's last move
        self.opponent_moves = set()  # Set to track opponent's moves

    def get_move(self, state: np.array) -> Tuple[int, int]:
        start_time = time.time()
        valid_moves = get_valid_actions(state)
        opponent_number = 2 if self.player_number == 1 else 1

        # Update opponent's moves
        self.update_opponent_moves(state)

        # Check if AI can win with this move
        for move in valid_moves:
            temp_state = np.copy(state)
            temp_state[move[0], move[1]] = self.player_number
            if check_win(temp_state, move, self.player_number)[0]:
                print(f"AI selects winning move: {move}")
                self.opponent_last_move = move
                return (int(move[0]), int(move[1]))

        # Check if opponent can win with the next move and block it
        for move in valid_moves:
            temp_state = np.copy(state)
            temp_state[move[0], move[1]] = opponent_number
            if check_win(temp_state, move, opponent_number)[0]:
                print(f"AI blocks opponent's winning move: {move}")
                self.opponent_last_move = move
                return (int(move[0]), int(move[1]))

        # Check if we can form the triple VC frame
        move = self.form_triple_vc(state)
        if move:
            print(f"AI forms triple VC frame at: {move}")
            self.opponent_last_move = move
            return (int(move[0]), int(move[1]))

        # Check if we need to block opponent's triple VC frame
        move = self.block_triple_vc(state)
        if move:
            print(f"AI blocks opponent's triple VC frame at: {move}")
            self.opponent_last_move = move
            return (int(move[0]), int(move[1]))

        # Look for 3-move combinations to block or win
        combo_move = self.lookahead_checkmate(state, valid_moves)
        if combo_move:
            print(f"AI selects strategic move: {combo_move}")
            self.opponent_last_move = combo_move
            return (int(combo_move[0]), int(combo_move[1]))

        # Update or create root node for MCTS
        if self.root is not None:
            self.root = self.update_tree_with_opponent_move(state)
        else:
            self.root = self.MCTSNode(state, None, player=self.player_number)

        # MCTS with adaptive simulations based on remaining time
        mcts_move = self.mcts_mix(state, valid_moves, start_time)
        if mcts_move in valid_moves:
            print(f"AI selects MCTS move: {mcts_move}")
            self.opponent_last_move = mcts_move
            return (int(mcts_move[0]), int(mcts_move[1]))
        else:
            # As a safety net, return a random valid move
            safe_move = random.choice(valid_moves)
            print(f"AI selects fallback move: {safe_move}")
            self.opponent_last_move = safe_move
            return (int(safe_move[0]), int(safe_move[1]))

    def update_opponent_moves(self, state):
        # Update the set of opponent's moves based on the current state
        opponent_number = 2 if self.player_number == 1 else 1
        opponent_positions = np.argwhere(state == opponent_number)
        self.opponent_moves = set(tuple(pos) for pos in opponent_positions)

    def get_triple_vc_patterns(self):
        # Hardcoded patterns for a size 4 board (7x7 grid)
        patterns = [
            [(0,0), (1,2), (0,3)],
            [(0,3), (2,4), (3,6)],
            [(3,6), (5,4), (6,3)],
            [(6,3), (5,1), (6,0)],
            [(6,0), (4,1), (3,0)],
            [(3,0), (1,2), (0,0)]
        ]
        return patterns

    def form_triple_vc(self, state):
        patterns = self.get_triple_vc_patterns()
        for pattern in patterns:
            occupied_positions = []
            empty_positions = []
            for pos in pattern:
                if 0 <= pos[0] < state.shape[0] and 0 <= pos[1] < state.shape[1]:
                    if state[pos[0], pos[1]] == self.player_number:
                        occupied_positions.append(pos)
                    elif state[pos[0], pos[1]] == 0:
                        empty_positions.append(pos)
                    else:
                        break  # Opponent occupies this position; cannot form this pattern
                else:
                    break  # Position out of bounds; skip this pattern
            else:
                # If we can occupy the empty positions to complete the pattern
                if len(empty_positions) > 0:
                    return empty_positions[0]  # Prioritize occupying the first empty position
        return None

    def block_triple_vc(self, state):
        opponent_number = 2 if self.player_number == 1 else 1
        patterns = self.get_triple_vc_patterns()
        for pattern in patterns:
            opponent_positions = []
            empty_positions = []
            for pos in pattern:
                if 0 <= pos[0] < state.shape[0] and 0 <= pos[1] < state.shape[1]:
                    if state[pos[0], pos[1]] == opponent_number:
                        opponent_positions.append(pos)
                    elif state[pos[0], pos[1]] == 0:
                        empty_positions.append(pos)
                    else:
                        break  # We occupy this position; opponent cannot form this pattern
                else:
                    break  # Position out of bounds; skip this pattern
            else:
                # If the opponent can occupy the empty positions to complete the pattern
                if len(opponent_positions) >= 2 and len(empty_positions) > 0:
                    return empty_positions[0]  # Block the opponent by occupying one of the empty positions
        return None

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

    def update_tree_with_opponent_move(self, state):
        for child in self.root.children:
            if np.array_equal(child.state, state):
                return child
        return self.MCTSNode(state, None, player=self.player_number)

    def mcts_mix(self, state: np.array, valid_moves: list, start_time: float) -> Tuple[int, int]:
        simulations = 0
        time_limit = self.time_per_move - 0.1  # Reserve a bit of time
        while time.time() - start_time < time_limit:
            # Decide whether to use heuristics or not
            use_heuristics = random.random() < 0.6  # 60% chance to use heuristics
            node, state_copy = self.select_node(self.root, use_heuristics)
            if not node.children:
                self.expand_node(node, node.player, use_heuristics)
            rewards = [self.rollout(state_copy, use_heuristics) for _ in range(self.num_rollouts_per_simulation)]
            average_reward = sum(rewards) / len(rewards)
            self.backpropagate(node, average_reward)
            simulations += 1

        print(f"Simulations performed: {simulations}")

        if self.root.children:
            # Select the move with the highest visit count
            best_child = max(self.root.children, key=lambda x: x.visits)
            return best_child.move

        # If no children (should not happen), return a random valid move
        return random.choice(valid_moves)

    def select_node(self, node, use_heuristics):
        state_copy = np.copy(node.state)
        while node.children:
            # Prioritize proven wins
            proven_wins = [child for child in node.children if child.is_terminal and child.is_win]
            if proven_wins:
                node = random.choice(proven_wins)
            else:
                # Avoid proven losses
                non_losing_children = [child for child in node.children if not (child.is_terminal and not child.is_win)]
                if non_losing_children:
                    node = self.ucb_select(node, non_losing_children, use_heuristics)
                else:
                    # All children are proven losses, select any child
                    node = random.choice(node.children)
            # Apply the move to the state copy
            if node.move is not None:
                state_copy[node.move[0], node.move[1]] = node.player
        return node, state_copy

    def expand_node(self, node, player_number, use_heuristics):
        opponent_number = 2 if player_number == 1 else 1
        valid_moves = get_valid_actions(node.state)

        # Check for immediate win
        for move in valid_moves:
            temp_state = np.copy(node.state)
            temp_state[move[0], move[1]] = player_number
            if check_win(temp_state, move, player_number)[0]:
                # Winning move found
                child_node = self.MCTSNode(temp_state, move, parent=node, player=opponent_number)
                child_node.is_terminal = True
                child_node.is_win = True
                node.children = [child_node]
                self.backup_proof(child_node)
                return

        # Check for opponent threats
        immediate_threats = []
        for move in valid_moves:
            temp_state = np.copy(node.state)
            temp_state[move[0], move[1]] = opponent_number
            if check_win(temp_state, move, opponent_number)[0]:
                immediate_threats.append(move)

        if len(immediate_threats) == 1:
            # Must block the threat
            move = immediate_threats[0]
            temp_state = np.copy(node.state)
            temp_state[move[0], move[1]] = player_number
            child_node = self.MCTSNode(temp_state, move, parent=node, player=opponent_number)
            node.children = [child_node]
            self.expand_node(child_node, opponent_number, use_heuristics)
        elif len(immediate_threats) >= 2:
            # Cannot block all threats, node is a loss
            node.is_terminal = True
            node.is_win = False
            self.backup_proof(node)
            return
        else:
            # No immediate win or loss, proceed normally
            for move in valid_moves:
                temp_state = np.copy(node.state)
                temp_state[move[0], move[1]] = player_number
                heuristic_value = 0
                if use_heuristics:
                    heuristic_value = self.evaluate_move_heuristic(temp_state, move, player_number)
                child_node = self.MCTSNode(temp_state, move, parent=node, player=opponent_number, heuristic_value=heuristic_value)
                node.children.append(child_node)

    def backup_proof(self, node):
        # Back up the proof to parent nodes
        while node.parent is not None:
            parent = node.parent
            if node.is_win:
                # If child is a win for current player, parent is a loss for opponent
                parent.is_terminal = True
                parent.is_win = False
            else:
                # If all children are losses, parent is a win
                if all(child.is_terminal and not child.is_win for child in parent.children):
                    parent.is_terminal = True
                    parent.is_win = True
                else:
                    break
            node = parent

    def ucb_select(self, node, children=None, use_heuristics=True):
        if children is None:
            children = node.children
        total_visits = sum(child.visits for child in children) + 1
        ucb_values = []
        for child in children:
            exploitation = (child.value / (child.visits + 1e-5))
            exploration = np.sqrt(np.log(total_visits) / (child.visits + 1e-5))
            heuristic_bias = 0
            if use_heuristics:
                heuristic_bias = (child.heuristic_value * self.heuristic_weight) / (child.visits + 1)
            ucb_value = exploitation + 2 * exploration + heuristic_bias
            ucb_values.append(ucb_value)
        max_index = np.argmax(ucb_values)
        return children[max_index]

    def rollout(self, state: np.array, use_heuristics: bool) -> float:
        # Create a copy of the state to avoid modifying the original
        state_copy = np.copy(state)
        current_player = self.player_number
        opponent_number = 2 if self.player_number == 1 else 1
        moves_played = []
        move_number = 0
        max_ring_depth = int(0.7 * (state_copy.size - np.count_nonzero(state_copy)))  # 70% of remaining moves
        while True:
            valid_moves = get_valid_actions(state_copy)
            if not valid_moves:
                break

            # Mate-in-one check for current player (first N moves)
            N = (state_copy.size - np.count_nonzero(state_copy)) // 2
            if move_number < N:
                for move in valid_moves:
                    temp_state = np.copy(state_copy)
                    temp_state[move[0], move[1]] = current_player
                    if check_win(temp_state, move, current_player)[0]:
                        state_copy[move[0], move[1]] = current_player
                        moves_played.append((current_player, move))
                        return 1.0 if current_player == self.player_number else 0.0

            if use_heuristics:
                # LGRF policy
                opponent_last_move = moves_played[-1][1] if moves_played else None
                lgrf_move = self.lgrf_table.get((opponent_last_move, current_player), None)
                if lgrf_move and lgrf_move in valid_moves:
                    move = lgrf_move
                else:
                    # Heuristic rollout policy
                    moves_with_heuristics = []
                    for move in valid_moves:
                        heuristic_value = self.evaluate_move_heuristic(state_copy, move, current_player)
                        moves_with_heuristics.append((move, heuristic_value))

                    # Select move with highest heuristic value
                    if moves_with_heuristics:
                        moves_with_heuristics.sort(key=lambda x: x[1], reverse=True)
                        top_moves = [m for m in moves_with_heuristics if m[1] == moves_with_heuristics[0][1]]
                        move = random.choice(top_moves)[0]
                    else:
                        move = random.choice(valid_moves)
            else:
                # Random rollout
                move = random.choice(valid_moves)

            state_copy[move[0], move[1]] = current_player
            moves_played.append((current_player, move))
            move_number += 1

            win_result = check_win(state_copy, move, current_player)
            if win_result[0]:
                if win_result[1] == 'ring':
                    if move_number <= max_ring_depth:
                        # Accept the ring win
                        # Update LGRF table
                        if use_heuristics and current_player == self.player_number:
                            for i in range(len(moves_played) - 2, -1, -2):
                                key = (moves_played[i][1], moves_played[i][0])
                                self.lgrf_table[key] = moves_played[i + 1][1]
                        return 1.0 if current_player == self.player_number else 0.0
                    else:
                        # Ignore the ring win, continue the rollout
                        pass
                else:
                    # Bridge or fork win
                    # Update LGRF table
                    if use_heuristics and current_player == self.player_number:
                        for i in range(len(moves_played) - 2, -1, -2):
                            key = (moves_played[i][1], moves_played[i][0])
                            self.lgrf_table[key] = moves_played[i + 1][1]
                    return 1.0 if current_player == self.player_number else 0.0

            # Switch player
            current_player = opponent_number if current_player == self.player_number else self.player_number
        return 0.5  # Return a neutral reward if no winner

    def backpropagate(self, node, reward: float):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def evaluate_move_heuristic(self, state, move, player_number):
        heuristic_value = 0
        dim = (state.shape[0] + 1) // 2
        opponent_number = 2 if player_number == 1 else 1

        # Do not consider moves that are already occupied
        if state[move[0], move[1]] != 0:
            return float('-inf')  # Invalid move; assign negative infinity to avoid selection

        # Heuristic 1: Locality (playing near own stones)
        own_stones = np.argwhere(state == player_number)
        if own_stones.size > 0:
            distances = np.abs(own_stones - move).sum(axis=1)
            min_distance = np.min(distances)
            if min_distance == 1:
                heuristic_value += 3  # Direct neighbor
            elif min_distance == 2:
                heuristic_value += 2  # Potential VC
            elif min_distance == 3:
                heuristic_value += 1  # Distance 2 but not VC

        # Heuristic 2: Edge Connectivity
        edge_bonus = 0
        group_edges = self.get_group_edges(state, move, player_number)
        edge_bonus += len(group_edges)
        heuristic_value += edge_bonus * 2  # Bonus per connected edge

        # Heuristic 3: Group Size
        group_size = len(self.get_connected_group(state, move, player_number))
        heuristic_value += group_size // 2  # Bonus for larger groups

        # Heuristic 4: Local Reply (playing near opponent's last move)
        if self.opponent_last_move is not None:
            distance_to_opponent = np.abs(np.array(self.opponent_last_move) - np.array(move)).sum()
            if distance_to_opponent == 1:
                heuristic_value += 3  # Direct neighbor
            elif distance_to_opponent == 2:
                heuristic_value += 2
            elif distance_to_opponent == 3:
                heuristic_value += 1

        # Heuristic 5: Virtual Connections Enhancement
        vc_bonus = self.evaluate_virtual_connection(state, move, player_number)
        heuristic_value += vc_bonus

        # Heuristic 6: Opponent Blocking VC
        if self.is_opponent_blocking_vc(state, move, player_number):
            heuristic_value -= 5  # Penalty for moves that allow the opponent to block VCs

        return heuristic_value

    def evaluate_virtual_connection(self, state, move, player_number):
        dim = (state.shape[0] + 1) // 2
        opponent_number = 2 if player_number == 1 else 1

        # Apply the move
        state_after = np.copy(state)
        state_after[move[0], move[1]] = player_number

        # Get connected components after the move
        components = self.get_player_connected_components(state_after, player_number)

        vc_bonus = 0
        for component in components:
            edges, corners = self.get_component_edges_corners(component, dim)
            connections = len(edges) + len(corners)

            # Check for potential winning conditions
            if connections >= 2:
                vc_bonus += 10  # Significant bonus for connecting multiple edges/corners

            if len(corners) >= 2:
                vc_bonus += 50  # High bonus for connecting two corners
            if len(edges) >= 3:
                vc_bonus += 50  # High bonus for connecting three edges

            # Bonus for moves that join existing VCs
            if self.joins_virtual_connections(state, move, player_number):
                vc_bonus += 30  # Assign a high bonus for joining VCs

            # General bonus for increasing VC set size
            vc_bonus += connections * 2  # Adjust the multiplier as needed

            # Subtract points if opponent's stones are blocking the component
            for pos in component:
                for neighbor in get_neighbours(dim, pos):
                    if tuple(neighbor) in self.opponent_moves:
                        vc_bonus -= 2  # Penalty for each opponent stone adjacent to the component

        return vc_bonus

    def joins_virtual_connections(self, state, move, player_number):
        # Determine if the move joins two or more of the player's existing VCs
        state_before = np.copy(state)
        components_before = self.get_player_connected_components(state_before, player_number)
        num_components_before = len(components_before)

        # Apply the move
        state_after = np.copy(state)
        state_after[move[0], move[1]] = player_number
        components_after = self.get_player_connected_components(state_after, player_number)
        num_components_after = len(components_after)

        # If the number of components decreases, the move has joined VCs
        return num_components_after < num_components_before

    def is_opponent_blocking_vc(self, state, move, player_number):
        opponent_number = 2 if player_number == 1 else 1
        dim = (state.shape[0] + 1) // 2

        # Compute player's VC set before the opponent's move
        components_before = self.get_player_connected_components(state, player_number)
        vc_set_before = set()
        for component in components_before:
            edges, corners = self.get_component_edges_corners(component, dim)
            vc_set_before.update(edges)
            vc_set_before.update(corners)
        vc_size_before = len(vc_set_before)

        # Apply the opponent's move
        state_after = np.copy(state)
        state_after[move[0], move[1]] = opponent_number

        # Compute player's VC set after the opponent's move
        components_after = self.get_player_connected_components(state_after, player_number)
        vc_set_after = set()
        for component in components_after:
            edges, corners = self.get_component_edges_corners(component, dim)
            vc_set_after.update(edges)
            vc_set_after.update(corners)
        vc_size_after = len(vc_set_after)

        # Determine if the opponent's move blocks the player's VC
        return vc_size_after < vc_size_before

    def get_player_connected_components(self, state, player_number):
        visited = set()
        components = []
        for row in range(state.shape[0]):
            for col in range(state.shape[1]):
                if state[row, col] == player_number and (row, col) not in visited:
                    component = set()
                    queue = deque()
                    queue.append((row, col))
                    visited.add((row, col))
                    component.add((row, col))
                    while queue:
                        pos = queue.popleft()
                        neighbors = get_neighbours((state.shape[0] + 1) // 2, pos)
                        for n in neighbors:
                            n_pos = (n[0], n[1])
                            if state[n[0], n[1]] == player_number and n_pos not in visited:
                                visited.add(n_pos)
                                component.add(n_pos)
                                queue.append(n_pos)
                    components.append(component)
        return components

    def get_component_edges_corners(self, component, dim):
        edges = set()
        corners = set()
        for pos in component:
            edge = get_edge(pos, dim)
            if edge != -1:
                edges.add(edge)
            corner = get_corner(pos, dim)
            if corner != -1:
                corners.add(corner)
        return edges, corners

    def get_group_edges(self, state, move, player_number):
        dim = (state.shape[0] + 1) // 2
        group = self.get_connected_group(state, move, player_number)
        edges_connected = set()
        for pos in group:
            edge = get_edge(pos, dim)
            if edge != -1:
                edges_connected.add(edge)
            corner = get_corner(pos, dim)
            if corner != -1:
                edges_connected.add(corner + 6)  # Distinguish corners
        return edges_connected

    def get_connected_group(self, state, start_pos, player_number):
        visited = set()
        queue = deque()
        queue.append(tuple(start_pos))
        visited.add(tuple(start_pos))
        while queue:
            pos = queue.popleft()
            neighbors = get_neighbours((state.shape[0] + 1) // 2, pos)
            for n in neighbors:
                n_pos = (n[0], n[1])
                if state[n[0], n[1]] == player_number and n_pos not in visited:
                    visited.add(n_pos)
                    queue.append(n_pos)
        return visited

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
            self.is_terminal = False
            self.is_win = False
