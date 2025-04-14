import random
# Used for speedup
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import gym
from gym import spaces


# Core game mechanics as standalone Numba functions
@njit
def compress_numba(row):
    """Compress row by shifting non-zero tiles to the left"""
    new_row = row[row != 0]
    result = np.zeros_like(row)
    result[:len(new_row)] = new_row
    return result

@njit
def merge_numba(row):
    """Merge identical adjacent tiles in a row"""
    score_increase = 0
    for i in range(len(row) - 1):
        if row[i] == row[i + 1] and row[i] != 0:
            row[i] *= 2
            row[i + 1] = 0
            score_increase += row[i]
    return row, score_increase

@njit
def move_left_numba(board):
    """Perform left move operation on the board"""
    size = board.shape[0]
    moved = False 
    score_increase = 0
    for i in range(size):
        original_row = board[i].copy()
        new_row = compress_numba(board[i])
        new_row, row_score = merge_numba(new_row)
        new_row = compress_numba(new_row)
        board[i] = new_row
        score_increase += row_score
        if not np.array_equal(original_row, new_row):
            moved = True
            
    return board, score_increase, moved

@njit
def move_right_numba(board):
    """Perform right move operation on the board"""
    size = board.shape[0]
    moved = False
    score_increase = 0
    for i in range(size):
        original_row = board[i].copy()
        reversed_row = np.flip(board[i])
        reversed_row = compress_numba(reversed_row)
        reversed_row, row_score = merge_numba(reversed_row)
        reversed_row = compress_numba(reversed_row)
        board[i] = np.flip(reversed_row)
        score_increase += row_score
        if not np.array_equal(original_row, board[i]):
            moved = True
            
    return board, score_increase, moved

@njit
def move_up_numba(board):
    """Perform up move operation on the board"""
    size = board.shape[0]
    moved = False
    score_increase = 0
    for j in range(size):
        original_col = board[:, j].copy()
        col = compress_numba(board[:, j])
        col, col_score = merge_numba(col)
        col = compress_numba(col)
        board[:, j] = col
        score_increase += col_score
        if not np.array_equal(original_col, board[:, j]):
            moved = True
            
    return board, score_increase, moved

@njit
def move_down_numba(board):
    """Perform down move operation on the board"""
    size = board.shape[0]
    moved = False
    score_increase = 0
    
    for j in range(size):
        original_col = board[:, j].copy()
        reversed_col = np.flip(board[:, j])
        reversed_col = compress_numba(reversed_col)
        reversed_col, col_score = merge_numba(reversed_col)
        reversed_col = compress_numba(reversed_col)
        board[:, j] = np.flip(reversed_col)
        score_increase += col_score
        if not np.array_equal(original_col, board[:, j]):
            moved = True      
    return board, score_increase, moved

@njit
def step_without_random_tile_numba(board, action):
    """Numba-optimized version of step_without_random_tile"""
    score_increase = 0
    moved = False
    
    if action == 0:
        board, score_increase, moved = move_up_numba(board)
    elif action == 1:
        board, score_increase, moved = move_down_numba(board)
    elif action == 2:
        board, score_increase, moved = move_left_numba(board)
    elif action == 3:
        board, score_increase, moved = move_right_numba(board)
        
    return board, score_increase, moved


COLOR_MAP = {
    0: "#cdc1b4", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179",
    16: "#f59563", 32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72",
    256: "#edcc61", 512: "#edc850", 1024: "#edc53f", 2048: "#edc22e",
    4096: "#3c3a32", 8192: "#3c3a32", 16384: "#3c3a32", 32768: "#3c3a32"
}
TEXT_COLOR = {
    2: "#776e65", 4: "#776e65", 8: "#f9f6f2", 16: "#f9f6f2",
    32: "#f9f6f2", 64: "#f9f6f2", 128: "#f9f6f2", 256: "#f9f6f2",
    512: "#f9f6f2", 1024: "#f9f6f2", 2048: "#f9f6f2", 4096: "#f9f6f2"
}

class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True

        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        # shifts all non-zero tiles to the left, and pad 0 to the right. 
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def merge(self, row):
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        if np.any(self.board == 0):
            return False
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        prev_score = self.score

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved

        if moved:
            self.add_random_tile()

        done = self.is_game_over()
        reward = self.score - prev_score # reward gain from this move only
        # return reward instead of self.score
        return self.board, reward, done, {}
        # return self.board, self.score, done, {}

    def step_without_random_tile(self, action):
        assert self.action_space.contains(action), "Invalid action"

        prev_score = self.score

        # Call the Numba-optimized function
        sim_board = self.board.copy()
        new_board, score_increase, moved = step_without_random_tile_numba(sim_board, action)

        # Update the environment state
        if moved:
            self.board = new_board
            self.score += score_increase

        self.last_move_valid = moved
        reward = score_increase  # reward is just the score gained in this move
        return self.board, reward, moved, {}

    def render(self, mode="human", action=None):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def simulate_row_move(self, row):
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")
        return not np.array_equal(self.board, temp_board)

