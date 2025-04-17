import random, copy
import math
import sys
import numpy as np
from tqdm import tqdm
from itertools import combinations
from typing import Union, Optional, List, Tuple
from numba import njit
from const import BLACK, WHITE, DIRECTIONS, BASE_PATTERNS, PLAYER_MAP
from utils import get_candidate_positions

@njit
def has_n_in_a_row_numba(board, size, color, n, directions):
    """
    Checks for a sequence of exactly n stones of a given color on the board.
    Uses four directions (horizontal, vertical, diagonal, anti-diagonal).
    """
    for r in range(size):
        for c in range(size):
            if board[r, c] != color:
                continue
            for d in range(directions.shape[0]):
                dr = directions[d, 0]
                dc = directions[d, 1]
                # Skip if the previous cell in the same direction is also of the same color.
                prev_r = r - dr
                prev_c = c - dc
                if 0 <= prev_r < size and 0 <= prev_c < size and board[prev_r, prev_c] == color:
                    continue
                count = 0
                rr = r
                cc = c
                for i in range(n):
                    if rr >= 0 and rr < size and cc >= 0 and cc < size and board[rr, cc] == color:
                        count += 1
                        rr += dr
                        cc += dc
                    else:
                        break
                if count == n:
                    return True
    return False

@njit
def pattern_score_numba(board, size, color, patterns, weights, directions):
    """
    Scans the board for every 6-cell segment in each given direction.
    For each segment, translates the cell values as:
      1  if the cell equals the given color,
      0  if the cell is empty,
     -1  if the cell is off-board or has the opponent's stone.
    When the segment exactly matches one of the known patterns (or its reverse),
    the corresponding weight is added to the total score.
    """
    score = 0.0
    num_patterns = patterns.shape[0]
    pat_len = patterns.shape[1]  # should be 6
    for r in range(size):
        for c in range(size):
            for d in range(directions.shape[0]):
                dr = directions[d, 0]
                dc = directions[d, 1]
                current_pattern = np.empty(pat_len, dtype=np.int64)
                for j in range(pat_len):
                    nr = r + dr * j
                    nc = c + dc * j
                    if nr < 0 or nr >= size or nc < 0 or nc >= size:
                        current_pattern[j] = -1
                    else:
                        cell = board[nr, nc]
                        if cell == color:
                            current_pattern[j] = 1
                        elif cell == 0:
                            current_pattern[j] = 0
                        else:
                            current_pattern[j] = -1
                            
                # Compare current_pattern to each stored pattern.
                for k in range(num_patterns):
                    match = True
                    for j in range(pat_len):
                        if current_pattern[j] != patterns[k, j]:
                            match = False
                            break
                    if match:
                        score += weights[k]
    return score


class Connect6Evaluator:
    def __init__(self, size: int = 19):
        self.size = size

        # Build patterns and weights.
        base_patterns = BASE_PATTERNS
        
        # Automatically add the reversed pattern if not already present.
        patterns_list = []
        weights_list = []
        for pat, w in base_patterns:
            if pat not in patterns_list:
                patterns_list.append(pat)
                weights_list.append(w)
                
            rev = pat[::-1]
            if rev not in patterns_list:
                patterns_list.append(rev)
                weights_list.append(w)
                
        # Convert the list of patterns and weights into NumPy arrays.
        self.patterns = np.array(patterns_list, dtype=np.int64)
        self.weights = np.array(weights_list, dtype=np.float64)
    
    def evaluate(self, board, player):
        """
        Evaluate the board state from the perspective of player.
        Returns a large positive value if player wins,
        a large negative value if the opponent wins,
        or a heuristic evaluation from the difference in pattern scores.
        """
        
        opponent = 3 - player
        
        # Check for an immediate win/loss.        
        if has_n_in_a_row_numba(board, self.size, player, 6, DIRECTIONS):
            return 1e6
        if has_n_in_a_row_numba(board, self.size, opponent, 6, DIRECTIONS):
            return -1e6

        my_score = pattern_score_numba(board, self.size, player,
                                       self.patterns, self.weights, DIRECTIONS)
        opp_score = pattern_score_numba(board, self.size, opponent,
                                        self.patterns, self.weights, DIRECTIONS)
        return my_score - opp_score

def create_state_from_board(board, size):
    state = {c: [] for c in "BW"}
    for r in range(size):
        for c in range(size):
            if board[r, c] == 1:
                state["B"].append(r * size + c)
            elif board[r, c] == 2:
                state["W"].append(r * size + c)
    
    return state

def create_board_from_state(state, size):
    board = np.zeros((size, size), dtype=int)
    for idx in state.get("B", []):
        r, c = divmod(idx, size)
        if 0 <= r < size and 0 <= c < size: # Basic bounds check
             board[r, c] = BLACK
    for idx in state.get("W", []):
        r, c = divmod(idx, size)
        if 0 <= r < size and 0 <= c < size: # Basic bounds check
             board[r, c] = WHITE
    return board


class UCTNode:
    def __init__(self, state, player, is_root, parent=None, action=None, size=19):
        self.size = size
        self.state = state
        self.action = action # action taken from parent to reach this node
        self.parent = parent
        self.visits = 0
        self.total_reward = 0
        self.children = {}
        self.is_root = is_root
        self.player = player # 1 for black, 2 for white
        self.opponent = 3 - player

        self.board_for_analysis = create_board_from_state(self.state, self.size)
        self.candidate_positions = self.get_candidate_children()

    def get_candidate_children(self, k=25):
        return get_candidate_positions(self.board_for_analysis, self.player, self.opponent, is_root=self.is_root, k=25)[::-1]
    
    def fully_expanded(self):
        return len(self.candidate_positions) == 0
        

class UCT_MCTS:
    def __init__(self, env, evaluator: Connect6Evaluator, iterations=50, exploration_constant=1.41, rollout_depth=10, player=WHITE, size=19):
        self.env = env
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.player = player # we ourself are black or white
        self.size = size
        self.evaluator = evaluator
        self.curr_max_value = float("-inf")
        self.curr_min_value = float("inf")
        
    def set_uct_mcts_root(self, first_or_second_move):
        self.count = first_or_second_move # used to track when to switch player
        self.first_or_second_move = first_or_second_move # root is first or second move

    def create_env_from_state(self, state, turn):
        new_env = copy.deepcopy(self.env)
        new_env.board = self.create_board_from_state(state)
        new_env.simulation_mode = True
        return new_env

    def create_board_from_state(self, state):
        board = np.zeros_like(self.env.board, dtype=int)
        for idx in state["B"]:
            r, c = divmod(idx, self.size)
            board[r, c] = BLACK
        for idx in state["W"]:
            r, c = divmod(idx, self.size)
            board[r, c] = WHITE
        return board
    
    def select_child(self, node: Optional[UCTNode]): 
        # Use the UCT formula: Q + c * sqrt(log(parent_visits)/child_visits) to select the child
        best_score = float("-inf")
        best_child = None
        for action, child in node.children.items():
            # # Special case handling for winning/blocking moves
            # if child.visits == float("inf"):
            #     return child  # Immediately select winning move
            
            if child.visits == 0:
                uct_score = float("inf")
            else:
                Q = child.total_reward / child.visits
                U = self.c * np.sqrt(np.log(node.visits / child.visits))
                
                # if np.isinf(node.visits) or np.isinf(child.visits):
                #     U = self.c
                # else:
                #     U = self.c * np.sqrt(np.log(node.visits / child.visits))
                # uct_score = Q + U
                if node.player == self.player: # max for myself
                    uct_score = Q + U
                else: # min for opponent
                    uct_score = -Q + U 
                    
            if (uct_score >= best_score and best_child == None) or (uct_score > best_score and best_child is not None):
                best_score = uct_score
                best_child = child

        # if best_child == None:
        #     print(f"error: best_child is None, children_length {len(node.children)}")
        return best_child
    
    def expand(self, node: Optional[UCTNode], turn):
        sim_env = self.create_env_from_state(node.state, turn)
        if node.candidate_positions:
            pos, val = node.candidate_positions.pop()

            # ex: action = ((3, 2), (4, 5)); (row, col) format
            # convert to C4, F5; (col, row) format; 1-based indexing for row number
            move_str = f"{sim_env.index_to_label(pos[1])}{pos[0]+1}"

            sim_env.play_move(PLAYER_MAP[self.player], move_str, "here1")

            child_state = create_state_from_board(sim_env.board, self.size)
            if self.count % 2 != 0 : # switch child's player when self.count is odd
                child_player = 3 - self.player
            else:
                child_player = self.player      
                   
            child_node = UCTNode(child_state, player=child_player, is_root=False, parent=node, action=pos)
            # set child.visit directly for special move
            # if val == float("inf") or val == float("-inf"):
            #     child_node.total_reward = val
            child_node.total_reward = val
            
            node.children[tuple(pos)] = child_node
            self.count += 1
            return child_node, sim_env
        
        # print(f"error: return nothing in expand", file=sys.stderr)
        return node, sim_env

    def rollout(self, sim_env, node: Optional[UCTNode], radius=6):
        curr_player = node.player
        curr_count = self.count
        last_action = node.action
        
        for _ in range(self.rollout_depth):
            if last_action == None:
                break
            last_r, last_c = last_action
            potential_moves = [
                (r, c)
                for r in range(max(0, last_r - radius), min(self.size, last_r + radius + 1))
                for c in range(max(0, last_c - radius), min(self.size, last_c + radius + 1))
                if sim_env.board[r, c] == 0
            ]         
            # potential_moves = [(r, c) for r in range(self.size) for c in range(self.size) if sim_env.board[r, c] == 0]
            if len(potential_moves) == 0:
                break
            
            move = random.choice(potential_moves)
            move_str = f"{sim_env.index_to_label(move[1])}{move[0]+1}"
            sim_env.play_move(PLAYER_MAP[curr_player], move_str, "here2")
            
            if curr_count % 2 != 0: # switch to next player if odd
                curr_player = 3 - curr_player
            curr_count += 1
            last_action = move # update last_action

        estimated_value = self.evaluator.evaluate(sim_env.board.copy(), self.player) # evaluate the board always from my perspective
        self.curr_max_value = max(self.curr_max_value, estimated_value)
        self.curr_min_value = min(self.curr_min_value, estimated_value)
        if self.curr_max_value == self.curr_min_value:
            normalized_value = 0.0
        else:
            normalized_value = 2 * (estimated_value - self.curr_min_value) / (self.curr_max_value - self.curr_min_value) - 1 # rescale to [-1, 1]
        # return normalized_value
        return estimated_value
    
    def backpropagate(self, node: Optional[UCTNode], reward):
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def run_simulation(self, root: Optional[UCTNode]):
        node = root
        turn = node.player

        # selection
        # print(f"h1", file=sys.stderr)
        while node.fully_expanded() and node.children:
            node = self.select_child(node)
            turn = node.player

        # expansion
        # print(f"h2", file=sys.stderr)
        node, sim_env = self.expand(node, turn)

        # rollout
        # print(f"h3", file=sys.stderr)
        reward = self.rollout(sim_env, node)

        # backprop
        # print(f"h4", file=sys.stderr)
        self.backpropagate(node, reward)

    def best_action_distribution(self, root: Optional[UCTNode]):
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action

        return best_action


class Connect6Game:
    def __init__(self, size=19, simulation_mode=False):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)  # 0: Empty, 1: Black, 2: White
        # self.turn = 1  # 1: Black, 2: White
        self.game_over = False
        self.simulation_mode = simulation_mode
        self.last_opponent_move = None
        self.first_or_second_move = 0 # 0: firstmove, 1: secondmove

    def reset_board(self):
        """Clears the board and resets the game."""
        self.board.fill(0)
        # self.turn = 1
        self.game_over = False
        print("= ", flush=True)

    def set_board_size(self, size):
        """Sets the board size and resets the game."""
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        # self.turn = 1
        self.game_over = False
        print("= ", flush=True)

    def check_win(self):
        """Checks if a player has won.
        Returns:
        0 - No winner yet
        1 - Black wins
        2 - White wins
        """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != 0:
                    current_color = self.board[r, c]
                    for dr, dc in directions:
                        prev_r, prev_c = r - dr, c - dc
                        if 0 <= prev_r < self.size and 0 <= prev_c < self.size and self.board[prev_r, prev_c] == current_color:
                            continue
                        count = 0
                        rr, cc = r, c
                        while 0 <= rr < self.size and 0 <= cc < self.size and self.board[rr, cc] == current_color:
                            count += 1
                            rr += dr
                            cc += dc
                        if count >= 6:
                            return current_color
        return 0
    
    def index_to_label(self, col):
        """Converts column index to letter (skipping 'I')."""
        return chr(ord('A') + col + (1 if col >= 8 else 0))  # Skips 'I'

    def label_to_index(self, col_char):
        """Converts letter to column index (accounting for missing 'I')."""
        col_char = col_char.upper()
        if col_char >= 'J':  # 'I' is skipped
            return ord(col_char) - ord('A') - 1
        else:
            return ord(col_char) - ord('A')
        
    def play_move(self, color, move, text):
        """Places stones and checks the game status."""
        if self.game_over:
            print("? Game over")
            return

        stones = move.split(',')
        positions = []

        for stone in stones:
            stone = stone.strip()
            if len(stone) < 2:
                print("? Invalid format")
                return
            col_char = stone[0].upper()
            if not col_char.isalpha():
                print("? Invalid format")
                return
            col = self.label_to_index(col_char)
            try:
                row = int(stone[1:]) - 1
            except ValueError:
                print("? Invalid format")
                return
            if not (0 <= row < self.size and 0 <= col < self.size):
                print("? Move out of board range")
                return
            if self.board[row, col] != 0:
                print(text)
                print(move)
                print("? Position already occupied")
                return
            positions.append((row, col))

        for row, col in positions:
            self.board[row, col] = 1 if color.upper() == 'B' else 2
            
        self.last_opponent_move = positions[-1]

        # self.turn = 3 - self.turn

        if not self.simulation_mode:
            print('= ', end='', flush=True)

    def generate_move(self, color, uct_mcts: Optional[UCT_MCTS]):
        """Generates a random move for the computer."""
        if self.game_over:
            print("? Game over")
            return

        if np.count_nonzero(self.board) == 0:
            empty_positions = [(r, c) for r in range(self.size) for c in range(self.size) if self.board[r, c] == 0]
            selected = random.sample(empty_positions, 1)
            move_str = ",".join(f"{self.index_to_label(c)}{r+1}" for r, c in selected)
            uct_mcts.player = BLACK # default to white, if we play the first move, then we're black.
            
        else:
            root_state = create_state_from_board(self.board, self.size)
            root = UCTNode(root_state, uct_mcts.player, is_root=True)

            uct_mcts.iterations = len(root.candidate_positions) * 200
            
            for _ in tqdm(range(uct_mcts.iterations)):
                uct_mcts.set_uct_mcts_root(self.first_or_second_move) # reset first_or_second_move before running each iteration
                uct_mcts.run_simulation(root)

            best_action = uct_mcts.best_action_distribution(root)
            move_str = f"{self.index_to_label(best_action[1])}{best_action[0]+1}"
            # move_str = ",".join(f"{self.index_to_label(c)}{r+1}" for r, c in best_action)
            
            self.first_or_second_move = (self.first_or_second_move + 1) % 2 # update


        self.play_move(color, move_str, "here3")

        print(f"{move_str}\n\n", end='', flush=True)
        print(move_str, file=sys.stderr)
        return
    
    def show_board(self):
        """Displays the board as text."""
        # print("= ")
        for row in range(self.size - 1, -1, -1):
            line = f"{row+1:2} " + " ".join("X" if self.board[row, col] == 1 else "O" if self.board[row, col] == 2 else "." for col in range(self.size))
            print(line)
        col_labels = "   " + " ".join(self.index_to_label(i) for i in range(self.size))
        print(col_labels)
        print(flush=True)

    def list_commands(self):
        """Lists all available commands."""
        print("= ", flush=True)  

    def process_command(self, command, uct_mcts):
        """Parses and executes GTP commands."""
        command = command.strip()
        if command == "get_conf_str env_board_size:":
            print(flush=True)
            return "env_board_size=19"

        if not command:
            return
        
        parts = command.split()
        cmd = parts[0].lower()

        if cmd == "boardsize":
            try:
                size = int(parts[1])
                self.set_board_size(size)
            except ValueError:
                print("? Invalid board size")
        elif cmd == "clear_board":
            self.reset_board()
        elif cmd == "play":
            if len(parts) < 3:
                print("? Invalid play command format")
            else:
                self.play_move(parts[1], parts[2], "here4")
                print('', flush=True)
        elif cmd == "genmove":
            if len(parts) < 2:
                print("? Invalid genmove command format")
            else:
                self.generate_move(parts[1], uct_mcts)
        elif cmd == "showboard":
            self.show_board()
        elif cmd == "list_commands":
            self.list_commands()
        elif cmd == "quit":
            print("= ", flush=True)
            sys.exit(0)
        else:
            print("? Unsupported command")

    def run(self, uct_mcts):
        """Main loop that reads GTP commands from standard input."""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                self.process_command(line, uct_mcts)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"? Error: {str(e)}")

if __name__ == "__main__":
    game = Connect6Game()
    evaluator = Connect6Evaluator()
    uct_mcts = UCT_MCTS(game, evaluator, iterations=100, exploration_constant=0.0, rollout_depth=10)
    game.run(uct_mcts)