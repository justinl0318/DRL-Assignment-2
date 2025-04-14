import copy
import random
import time
import math
import numpy as np
import pickle
from collections import defaultdict
from game_env import Game2048Env, step_without_random_tile_numba
from typing import Optional, Union

def rot90(coord, size):
    r, c = coord
    return (c, size - 1 - r)

def rot180(coord, size):
    r, c = coord
    return (size - 1 - r, size - 1 - c)

def rot270(coord, size):
    r, c = coord
    return (size - 1 - c, r)

def reflect_horizontal(coord, size):
    r, c = coord
    return (r, size - 1 - c)

def identity(coord, size):
    return coord

class NTupleApproximator:
    def __init__(self, board_size, patterns):
        """
        Initializes the N-Tuple approximator.
        Hint: you can adjust these if you want
        """
        self.board_size = board_size
        self.patterns = patterns
        # Create a weight dictionary for each pattern (shared within a pattern group)
        self.weights = [defaultdict(float) for _ in patterns]
        # Generate symmetrical transformations for each pattern
        self.symmetry_patterns = []
        self.symmetry_patterns_to_index = []
        for i, pattern in enumerate(self.patterns):
            syms = self.generate_symmetries(pattern)
            # print(syms)
            for syms_ in syms:
                self.symmetry_patterns.append(syms_)
                self.symmetry_patterns_to_index.append(i)
        self.symmetry_patterns_len = len(self.symmetry_patterns) 
        # print(self.symmetry_patterns_to_index)

    def generate_symmetries(self, pattern):
        # TODO: Generate 8 symmetrical transformations of the given pattern.
        rotations = [pattern]
        for _ in range(3):
            pattern = [(y, self.board_size - 1 - x) for x, y in pattern] # rotate clockwise 90°
            rotations.append(pattern)

        reflections = []
        for r in rotations:
            # horizontal flip:
            # a b -> b a
            # c d    d c
            reflected = [(x, self.board_size - 1 - y) for x, y in r] # rotations + 1-axis flip = complete symmetry group (Dihedral group D4)
            reflections.append(reflected)

        return rotations + reflections

    def tile_to_index(self, tile):
        """
        Converts tile values to an index for the lookup table.
        """
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board, coords):
        # TODO: Extract tile values from the board based on the given coordinates and convert them into a feature tuple.
        idx = 0
        for i, (x, y) in enumerate(coords):
            tile = self.tile_to_index(board[x][y])
            idx |= tile << (4 * i)
        return idx

    def value(self, board):
        # TODO: Estimate the board value: sum the evaluations from all patterns.
        val = 0
        for i, pattern in enumerate(self.symmetry_patterns):
            feature = self.get_feature(board, pattern)
            base_idx = self.symmetry_patterns_to_index[i]
            val += self.weights[base_idx][feature]
        return val

    def update(self, board, delta, alpha):
        # TODO: Update weights based on the TD error.
        for i, pattern in enumerate(self.symmetry_patterns):
            feature = self.get_feature(board, pattern)
            base_idx = self.symmetry_patterns_to_index[i]
            weight_update = (alpha * delta) / self.symmetry_patterns_len
            # print(f"Pattern: {pattern}, Feature: {feature}, Weight Update: {weight_update}")
            self.weights[base_idx][feature] += weight_update