# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import copy
import random
import math
import struct
from collections import defaultdict

from approximator import NTupleApproximator
from game_env import Game2048Env
from td_mcts import Decision_Node, Chance_Node, TD_MCTS

def load_cpp_ntuple_weights(filename, approximator):
    """
    Load C++ weights into a NTupleApproximator instance.
    
    The binary file format is assumed to be:
      - 8 bytes (size_t, little endian): number of features (feature_count)
      Then, for each feature:
        - 4 bytes (int, little endian): feature name length
        - (name length) bytes: feature name (UTF-8 encoded)
        - 8 bytes (size_t, little endian): number of weights for this feature
        - (number of weights * 4) bytes: weight array (floats, little endian)
    
    The approximator.weights is a list of defaultdict(float) with one entry per feature.
    """
    
    with open(filename, "rb") as f:
        # Read feature count (size_t, 8 bytes)
        feature_count_bytes = f.read(8)
        if len(feature_count_bytes) < 8:
            raise ValueError("File too short to read the feature count!")
        feature_count = struct.unpack("<Q", feature_count_bytes)[0]
        print(f"Total features in file: {feature_count}")

        if feature_count != len(approximator.weights):
            print(f"Warning: Feature count mismatch! "
                  f"Approximator expects {len(approximator.weights)} features, "
                  f"but file contains {feature_count}.")
        
        # For each feature, load the weight array and store into the respective dictionary.
        for i in range(feature_count):
            # Read the feature name length (4 bytes)
            name_len_bytes = f.read(4)
            if len(name_len_bytes) < 4:
                raise ValueError("Unexpected end-of-file reading feature name length.")
            name_len = struct.unpack("<i", name_len_bytes)[0]
            
            # Read the feature name (name_len bytes, UTF-8 encoded)
            name_bytes = f.read(name_len)
            if len(name_bytes) < name_len:
                raise ValueError("Unexpected end-of-file reading feature name.")
            feature_name = name_bytes.decode("utf-8")
            
            # Read the number of weights (size_t, 8 bytes)
            size_bytes = f.read(8)
            if len(size_bytes) < 8:
                raise ValueError("Unexpected end-of-file reading weight count.")
            feature_size = struct.unpack("<Q", size_bytes)[0]
            
            # Read the weight values: feature_size floats (4 bytes each)
            weights_bytes = f.read(feature_size * 4)
            if len(weights_bytes) < feature_size * 4:
                raise ValueError("Unexpected end-of-file reading weight data.")
            fmt = "<" + "f" * feature_size  # little-endian floats
            weight_values = struct.unpack(fmt, weights_bytes)
            
            # Convert the flat weight array into a dictionary
            # (e.g., key = index, value = weight)
            # You may choose to store only nonzero weights if you like.
            weight_dict = defaultdict(float, {j: weight for j, weight in enumerate(weight_values)})
            
            # Store into the approximator (each feature corresponds to one entry in the weights list)
            approximator.weights[i] = weight_dict
            print(f"Loaded feature {i}: '{feature_name}' with {feature_size} weights.")
    
patterns = [
    [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
    [(1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1)],
    [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (2, 2)],
    [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1), (2, 2)],
    [(0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (3, 2)],
    [(0, 0), (0, 1), (1, 1), (2, 0), (2, 1), (3, 1)],
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 2)]
]
approximator = NTupleApproximator(board_size=4, patterns=patterns)
weight_filename = "2048_300000.bin"
try:
    load_cpp_ntuple_weights(weight_filename, approximator)
    print("\nSuccessfully loaded C++ weights into the NTupleApproximator instance.")
except Exception as e:
    print(f"Error loading weights: {e}")

def get_action(state, score):
    global approximator

    env = Game2048Env()
    env.board = np.array(state, dtype=int)
    env.score = score
    legal_moves = [a for a in range(4) if env.is_move_legal(a)]
    if not legal_moves:
        return 0
    
    root = Decision_Node(state=env.board.copy(), score=env.score)
    td_mcts = TD_MCTS(env, approximator, iterations=50, exploration_constant=0.0, rollout_depth=0, gamma=0.99)

    for _ in range(td_mcts.iterations):
        td_mcts.run_simulation(root)

    best_action, distribution = td_mcts.best_action_distribution(root)
    return best_action