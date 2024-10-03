import random
import numpy as np
from typing import Tuple
from helper import *
import copy

class DSU:
    def __init__(self):
        self.parent = {}
        self.rank = {}
        self.dimension = 0
    def find(self, node):
        if node not in self.parent:
            self.parent[node] = node
            self.rank[node] = 0
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        return self.parent[node]
    def union(self, node1, node2):
        root1 = self.find(node1)
        root2 = self.find(node2)
        if root1 != root2:
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            elif self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1
    def connected(self, node1, node2):
        if node1 not in self.parent or node2 not in self.parent:
            return False
        return self.find(node1) == self.find(node2)
    def get_sets(self):
        from collections import defaultdict
        sets = defaultdict(set)
        for node in self.parent:
            root = self.find(node)
            sets[root].add(node)
        return dict(sets)
    def check_v_pairs(self, v_new, v_pair, v_n1, v_n2, state):
        # print("checking v pairs: ", v_new, v_pair, v_n1, v_n2)
        if state[v_new] == state[v_pair] and state[v_n1] == 0 and state[v_n2] == 0:
            return True
        return False 
    def insert_node(self, v_new, map_v_pairs, state):
        # print("inserting node: ", v_new)
        self.find(v_new)
        flag_for_virtual_cc = False
        for v in map_v_pairs.get(v_new, []):
            v2 = v[0]
            if self.check_v_pairs(v_new, v[0], v[1], v[2], state):
                self.union(v_new, v2)
                flag_for_virtual_cc = True
        for v in get_neighbours(self.dimension, v_new):
            if state[v[0], v[1]] == state[v_new[0], v_new[1]]:
                self.union(v, v_new)
        return flag_for_virtual_cc
    def re_evaluate(self, u, v, state, map_v_pairs):
        # print("re-evaluating: ", u, v)
        sets = self.get_sets()
        component = sets[self.find(u)]
        for node in component:
            self.parent[node] = node
            self.rank[node] = 0 
        for node in component:
            self.insert_node(node, map_v_pairs, state)   
    def recheck_nodes(self, move, inverse_map_v_pair, map_v_pairs, state): 
        # print("rechecking nodes: ", move)
        for (u,v) in inverse_map_v_pair.get(move, []):
            if self.connected(u, v):
                print("connected: ", u, v)
                self.re_evaluate(u, v, state, map_v_pairs)
    def copy(self):
        new_dsu = DSU()
        new_dsu.parent = copy.deepcopy(self.parent)
        new_dsu.rank = copy.deepcopy(self.rank)
        new_dsu.dimension = self.dimension  # Assuming dimension is an integer
        return new_dsu


class dsus:
    def __init__(self):
        self.player_dsu = DSU()
        self.opponent_dsu = DSU()

class AIPlayer:
    def __init__(self, player_number: int, timer, max_depth: int = 3, heuristic_weight=0.5, 
                 C: float = 0.00, TARGET_COOLDOWN: int = 3, ROLLOUT_DEPTH: int = 6, SIMULATIONS: int = 500):
        self.player_number = player_number
        self.opponent_number = 2 if player_number == 1 else 1
        self.type = 'ai2'
        self.player_string = f'Player {player_number}: ai2'
        self.timer = timer
        self.max_depth = max_depth
        self.heuristic_weight = heuristic_weight
        self.first_run = True
        self.map_v_pairs = {}
        self.inverse_map_v_pairs = {}
        self.virtual_cc = set(set())
        self.dsus = dsus()
        self.dimension = 0
        self.last_state = None
        self.last_opp_move = None
        self.edges = {}
        self.corners = set()
        self.bias_vector  = []
        self.biased_moves = []
        self.target_locked = False
        self.default_target_cooldown = TARGET_COOLDOWN
        self.default_simulations = SIMULATIONS
        self.default_rollout_depth = ROLLOUT_DEPTH
        self.target_cooldown = TARGET_COOLDOWN
        self.simulations = SIMULATIONS
        self.rollout_depth = ROLLOUT_DEPTH
        self.C = C

    def get_move(self, state: np.array) -> Tuple[int, int]:
        if self.first_run: 
            # Initialize for specific cases here: 
            self.dimension = len(state)
            if self.dimension == 7: 
                print("Customized simulations")
                self.default_simulations = 500
                self.simulations = self.default_simulations
                self.default_rollout_depth = 6
                self.rollout_depth = self.default_rollout_depth
            self.corners = set(get_all_corners(len(state)))
            sides = get_all_edges(len(state))
            for i, side in enumerate(sides):
                for coord in side:
                    self.edges[coord] = i + 1
            self.dsus.player_dsu.dimension = self.dimension
            self.dsus.opponent_dsu.dimension = self.dimension
            self.bias_vector = np.ones((self.dimension, self.dimension))
            self.reset_bias_vector(state)
            for r in range(len(state)):
                for c in range(len(state)):
                    if state[r,c] != 3:
                        n = self.get_v_pairs(state, (r,c))
                        self.map_v_pairs[(r,c)] = n
                        for v in n: 
                            if (v[0][0] > r or (v[0][0] == r and v[0][1] > c)):
                                if v[1] not in self.inverse_map_v_pairs:
                                    self.inverse_map_v_pairs[v[1]] = []
                                if v[2] not in self.inverse_map_v_pairs:
                                    self.inverse_map_v_pairs[v[2]] = []
                                self.inverse_map_v_pairs[v[1]].append(((r,c),v[0]))
                                self.inverse_map_v_pairs[v[2]].append(((r,c),v[0]))
            self.first_run = False

        if self.target_locked == True:
                self.simulations = 500
                print("Target is locked")
                if self.target_cooldown == 0:
                    self.target_locked = False
                    self.simulations = self.default_simulations
                    self.target_cooldown = self.default_target_cooldown
                    self.biased_moves = []
                    print("Target is unlocked")
                else:
                    self.target_cooldown -= 1
                    print(f"Target is locked for {self.target_cooldown} more turns")

        if self.last_state is not None:
            for r in range(len(state)):
                for c in range(len(state[r])):
                    if self.last_state[r, c] != state[r, c] and state[r, c] == self.opponent_number:
                        self.last_opp_move = (r, c)
                        self.dsus.opponent_dsu.insert_node((r, c), self.map_v_pairs, state)
                        self.dsus.player_dsu.recheck_nodes((r, c), self.inverse_map_v_pairs, self.map_v_pairs, state)
                        break
    
        def returning_chores(move, state):
            self.last_state = state.copy()
            temp_state = np.copy(state)
            temp_state[move[0], move[1]] = self.player_number
            self.dsus.player_dsu.insert_node(move, self.map_v_pairs, temp_state)
            self.dsus.opponent_dsu.recheck_nodes(move, self.inverse_map_v_pairs, self.map_v_pairs, temp_state)

        print("dsu_player is: ", self.dsus.player_dsu.get_sets())
        print("dsu_opponent is: ", self.dsus.opponent_dsu.get_sets())
        valid_moves = get_valid_actions(state)
        for move in valid_moves:
            temp_state = np.copy(state)
            temp_state[move[0], move[1]] = self.player_number
            if check_win(temp_state, move, self.player_number)[0]:
                print(f"AI selects winning move: {move}")
                returning_chores(move, state)
                return (int(move[0]), int(move[1]))

        # Check if opponent can win with the next move and block it
        opponent_number = 2 if self.player_number == 1 else 1
        for move in valid_moves:
            temp_state = np.copy(state)
            temp_state[move[0], move[1]] = opponent_number
            if check_win(temp_state, move, opponent_number)[0]:
                print(f"AI blocks opponent's winning move: {move}")
                returning_chores(move, state)
                return (int(move[0]), int(move[1]))

        # Look for 3-move combinations to block or win
        # combo_move = self.lookahead_checkmate(state, valid_moves)
        # if combo_move:
        #     print(f"AI selects strategic move: {combo_move}")
        #     returning_chores(combo_move, state)
        #     return (int(combo_move[0]), int(combo_move[1]))

        # Fallback to MCTS RAVE with heuristics if no immediate threats or wins are detected
        current_turn = np.count_nonzero(state)
        mcts_move = self.mcts_rave(state, valid_moves, current_turn)
        if mcts_move in valid_moves:
            print(f"AI selects MCTS move: {mcts_move}")
            returning_chores(mcts_move, state)
            return (int(mcts_move[0]), int(mcts_move[1]))
        else:
            # As a safety net, return a random valid move
            safe_move = random.choice(valid_moves)
            print(f"AI selects fallback move: {safe_move}")
            returning_chores(safe_move, state)
            return (int(safe_move[0]), int(safe_move[1]))

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

        self.simulations = self.default_simulations
        for _ in range(self.simulations):
            # print(self.simulations)
            node, state_copy = self.select_node(root)
            reward = self.rollout(state_copy)
            self.backpropagate(node, reward)

        if root.children:
            # Select the move with the highest visit count
            best_child = max(root.children, key=lambda x: x.visits)
            # root.print_mcts_tree(root)
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
            heuristic_value = self.evaluate_move_heuristic(temp_state, move, player_number, True)
            child_node = self.MCTSNode(temp_state, move, parent=node, player=opponent_number, heuristic_value=heuristic_value)
            node.children.append(child_node)

    def ucb_select(self, node):
        total_visits = sum(child.visits for child in node.children) + 1
        ucb_values = []
        for child in node.children:
            exploitation = (child.value / (child.visits + 1e-5))
            exploration = self.C * np.sqrt(np.log(total_visits) / (child.visits + 1e-5))
            heuristic_bias = (child.heuristic_value * self.heuristic_weight) / np.sqrt(child.visits + 1)
            ucb_value = exploitation + exploration + heuristic_bias
            ucb_values.append(ucb_value)
        max_index = np.argmax(ucb_values)
        return node.children[max_index]

    def rollout(self, state: np.array) -> float:
        state_copy = np.copy(state)
        current_player = self.player_number
        opponent_number = 2 if self.player_number == 1 else 1
        for _ in range(self.rollout_depth):  # Limit the rollout depth
            valid_moves = get_valid_actions(state_copy)
            if not valid_moves:
                break
            moves_with_heuristics = []
            for move in valid_moves:
                temp_state = np.copy(state_copy)
                temp_state[move[0], move[1]] = current_player
                heuristic_value = self.evaluate_move_heuristic(temp_state, move, current_player)
                moves_with_heuristics.append((move, heuristic_value))

            if moves_with_heuristics:
                moves_with_heuristics.sort(key=lambda x: x[1], reverse=True)
                half_index = len(moves_with_heuristics) // 2
                top_half_moves = moves_with_heuristics[:half_index]
                # top_half_moves = moves_with_heuristics
                move = random.choice(top_half_moves)[0]
            else:
                move = random.choice(valid_moves)
            state_copy[move[0], move[1]] = current_player
            if check_win(state_copy, move, current_player)[0]:
                return 1.0 if current_player == self.player_number else 0.0
            current_player = opponent_number if current_player == self.player_number else self.player_number
        return 0.5

    def backpropagate(self, node, reward: float):
        while node is not None:
            node.visits += 1
            if node.player == self.player_number:
                node.value += reward
            else:
                node.value += (1 - reward)
                # node.value += reward
            node = node.parent


#___________________________________________________________________________________________
    def evaluate_move_heuristic(self, state, move, player_number, is_expansion=False):
        heuristic_value = 0
        is_virtual = False
        blocks_virtual = False
        if is_expansion:
            # Making virtual Connections: ___________________________________________________________
            temp_player_dsu = self.dsus.player_dsu.copy()
            temp_opponent_dsu = self.dsus.opponent_dsu.copy()
            prev_player_sets = temp_player_dsu.get_sets()
            prev_opponent_sets = temp_opponent_dsu.get_sets()
            flag_for_virtual = temp_player_dsu.insert_node(move, self.map_v_pairs, state)
            temp_opponent_dsu.recheck_nodes(move, self.inverse_map_v_pairs, self.map_v_pairs, state)
            new_player_sets = temp_player_dsu.get_sets()
            new_opponent_sets = temp_opponent_dsu.get_sets()
            if len(prev_player_sets) >= len(new_player_sets) and flag_for_virtual:
                heuristic_value += 20
                is_virtual = True
            if len(prev_opponent_sets) < len(new_opponent_sets):
                heuristic_value += 5
                blocks_virtual = True

            # Pursuing depth 1 virtual connections:_____________________________________________________
            if not self.target_locked:
                for index in new_player_sets:
                    corner_count = 0
                    anchor_points = set()
                    for node in new_player_sets[index]:
                        # print("node haiss: ", node)
                        if node in self.corners:
                            corner_count += 1
                            anchor_points.add(node)
                    if corner_count >= 2:
                        self.target_locked = True
                        heuristic_value += 10000
                        print("Supreme W by corners....")
                        print("anchor_points: ", anchor_points)
                        for node in new_player_sets[index]:
                            for n in get_neighbours(self.dimension, node):
                                if state[n[0], n[1]] != 1:
                                    print("added biased point, ", n)
                                    self.biased_moves.append(n)
                        break
            #check frame for edges:
                for index in new_player_sets: 
                    edge_count = 0
                    seen_edges = set()
                    anchor_points = set()
                    for node in new_player_sets[index]:
                        # print("node hai: ", node)
                        node_edge = get_edge(node, self.dimension)
                        if node in self.edges and node_edge not in seen_edges:
                            edge_count += 1
                            anchor_points.add(node)
                            seen_edges.add(node_edge)
                    if edge_count >= 3:
                        self.target_locked = True
                        heuristic_value += 10000
                        print("Supreme W by edges....")
                        print("anchor_points: ", anchor_points)
                        for node in new_player_sets[index]: 
                            for n in get_neighbours(self.dimension, node):
                                if state[n[0], n[1]] != 3:
                                    self.biased_moves.append(n)
                        break
            
            # print("biased moves are: ", self.biased_moves)
            # print("move is : ", move)
            move_counts = {}
            # if move in self.biased_moves:
            #     heuristic_value += 200
            
            for b in self.biased_moves:
                if b in move_counts:
                    move_counts[b] += 1
                else:
                    move_counts[b] = 1
            for b, count in move_counts.items():
                if count >= 2 and move == b:
                    heuristic_value += 1000
            
            # Preventing depth 1 virtual connections:_____________________________________________________
            temp_state = np.copy(state)
            temp_state[move[0], move[1]] = 2
            temp_player_dsu = self.dsus.player_dsu.copy()
            temp_opponent_dsu = self.dsus.opponent_dsu.copy()
            temp_opponent_dsu.insert_node(move, self.map_v_pairs, temp_state)
            temp_player_dsu.recheck_nodes(move, self.inverse_map_v_pairs, self.map_v_pairs, temp_state)
            new_opponent_sets = temp_opponent_dsu.get_sets()

            for index in new_opponent_sets:
                corner_count = 0
                for node in new_opponent_sets[index]:
                    if node in self.corners:
                        corner_count += 1
                if corner_count >= 2:
                    heuristic_value += 5000
                    print("Preventing W by corners....")
                    break
            #check frame for edges:
            for index in new_opponent_sets:
                edge_count = 0
                seen_edges = set()
                for node in new_opponent_sets[index]:
                    node_edge = get_edge(node, self.dimension)
                    if node in self.edges and node_edge not in seen_edges:
                        edge_count += 1
                        seen_edges.add(node_edge)
                if edge_count >= 3:
                    heuristic_value += 5000
                    print("Preventing W by edges....")
                    break
            #Biasing towards the corner: 
            # if self.dimension > 7: 
            #     heuristic_value -= 2*(self.bias_vector[move[0]][move[1]])

        if move in self.edges:
            heuristic_value += 2
        if move in self.corners: 
            heuristic_value += 2
        # Heuristic 3: Group Size (approximate)
        # Bonus for connecting to own stones
        neighbors = get_neighbours(self.dimension, move)
        own_neighbors = [n for n in neighbors if state[n[0], n[1]] == player_number]
        heuristic_value += len(own_neighbors)  # Bonus for each own neighbor
        return heuristic_value
    #___________________________________________________________________________________________
    def set_anchor_point(self, anchor, factor=0.97):
        self.bias_vector[anchor] = 1
        frontier = set()
        frontier.add(anchor)
        new_frontier = set()
        while frontier:
            for node in frontier:
                for n in get_neighbours(self.dimension, node):
                    if self.bias_vector[n] == 0:
                        self.bias_vector[n] = 1*factor
                        new_frontier.add(n)
            frontier = new_frontier
            factor *= factor

    def reset_bias_vector(self, state, factor=0.97):
        self.bias_vector = np.ones((self.dimension, self.dimension))
        for r in range(self.dimension):
            for c in range(self.dimension):
                if state is not None and state[r, c] != 3:
                    for corner in self.corners:
                        distance = abs(r - corner[0]) + abs(c - corner[1])
                        self.bias_vector[r, c] = min((float(distance)  / self.dimension), self.bias_vector[r, c])
        print("Bias Vector has been reset")
        print(self.bias_vector)

    def get_v_pairs(self, state, vertex):
        i,j = vertex
        siz = len(state)//2
        dim = len(state)
        neighbors = []
        if j < siz-1:
            neighbors.append(((i-2,j-1),(i-1,j),(i-1,j-1)))
            neighbors.append(((i-1,j-2),(i-1,j-1),(i,j-1)))
            neighbors.append(((i+1,j-1),(i,j-1),(i+1,j)))
            neighbors.append(((i+2,j+1),(i+1,j),(i+1,j+1)))
            neighbors.append(((i+1,j+2),(i+1,j+1),(i,j+1)))
            neighbors.append(((i-1,j+1),(i,j+1),(i-1,j)))
        elif j == siz-1: 
            neighbors.append(((i-2,j-1), (i-1,j-1), (i-1,j)))
            neighbors.append(((i-1,j-2),(i-1,j-1),(i,j-1)))
            neighbors.append(((i+1,j-1),(i,j-1),(i+1,j)))
            neighbors.append(((i+2,j+1),(i+1,j),(i+1,j+1)))
            neighbors.append(((i,j+2),(i,j+1),(i+1,j+1)))
            neighbors.append(((i-1,j+1),(i,j+1),(i-1,j)))
        elif j == siz: 
            neighbors.append(((i-2,j-1), (i-1,j-1), (i-1,j)))
            neighbors.append(((i-1,j-2),(i-1,j-1),(i,j-1)))
            neighbors.append(((i+1,j-1),(i,j-1),(i+1,j)))
            neighbors.append(((i+1,j+1),(i+1,j),(i,j+1)))
            neighbors.append(((i-1,j+2),(i-1,j+1),(i,j+1)))
            neighbors.append(((i-2,j+1),(i-1,j),(i-1,j+1)))
        elif j == siz+1:
            neighbors.append(((i-1,j-1),(i,j-1),(i-1,j)))
            neighbors.append(((i,j-2),(i,j-1),(i+1,j-1)))
            neighbors.append(((i+2,j-1),(i+1,j),(i+1,j-1)))
            neighbors.append(((i+1,j+1),(i+1,j),(i,j+1)))
            neighbors.append(((i-1,j+2),(i-1,j+1),(i,j+1)))
            neighbors.append(((i-2,j+1),(i-1,j),(i-1,j+1)))
        else:
            neighbors.append(((i-1,j-1),(i,j-1),(i-1,j)))
            neighbors.append(((i+1,j-2),(i+1,j-1),(i,j-1)))
            neighbors.append(((i+2,j-1),(i+1,j),(i+1,j-1)))
            neighbors.append(((i+1,j+1),(i+1,j),(i,j+1)))
            neighbors.append(((i-1,j+2),(i-1,j+1),(i,j+1)))
            neighbors.append(((i-2,j+1),(i-1,j),(i-1,j+1)))
        
        valid_neighbors = [] 
        for n in neighbors: 
            (r,c) = n[0]
            if r < 0 or r > dim-1 or c < 0 or c > dim-1:
                continue
            if state[r,c] == 3:
                continue
            valid_neighbors.append(n)
        return valid_neighbors

    class MCTSNode:
        def __init__(self, state, move, parent=None, player=1, heuristic_value=0):
            self.state = state
            self.move = move
            self.parent = parent
            self.children = []
            self.visits = 0
            self.value = 0.0
            self.player = player
            self.heuristic_value = heuristic_value

        def print_mcts_tree(self, node, depth=0):
            indent = '  ' * depth
            print(f"{indent}Node at depth {depth}:")
            print(f"{indent}  Move: {node.move}")
            print(f"{indent}  Player: {node.player}")
            print(f"{indent}  Visits: {node.visits}")
            print(f"{indent}  Value: {node.value}")
            print(f"{indent}  Heuristic Value: {node.heuristic_value}")
            print(f"{indent}  Children count: {len(node.children)}")
            
            # Print the state's array representation (optional, for large states this could be omitted)
            print(f"{indent}  State: \n{node.state}")
            
            # Recursively print all children
            for child in node.children:
                self.print_mcts_tree(child, depth + 1)
