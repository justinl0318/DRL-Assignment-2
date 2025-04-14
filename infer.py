import pickle, random
import copy
import numpy as np
import struct
from collections import defaultdict
from td_mcts import TD_MCTS, Decision_Node, Chance_Node
from game_env import Game2048Env
from approximator import NTupleApproximator

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

env = Game2048Env()

# ----- Simulate n-tuple value approximator score -----
# # Initialize the game environment
# state = env.reset()
# env.render()
# done = False

# while not done:

#     legal_moves = [a for a in range(4) if env.is_move_legal(a)]
#     if not legal_moves:
#         break

#     # TODO: Use your N-Tuple approximator to play 2048
#     best_value = float("-inf")
#     best_action = None
#     for a in legal_moves:
#         sim_env = copy.deepcopy(env)
#         _, reward, _, _ = sim_env.step_without_random_tile(a)
#         value = reward + approximator.value(sim_env.board)
#         if value > best_value:
#             best_value = value
#             best_action = a

#     state, reward, done, _ = env.step(best_action)  # Apply the selected best_action
#     # env.render(action=best_action)  # Display the updated game state

# # Print final game results
# env.render(action=best_action)
# print("Game over, final score:", env.score)




# ----- Simulate TD-MCTS score -----
state = env.reset()
env.render()

done = False
while not done:
    legal_moves = [a for a in range(4) if env.is_move_legal(a)]
    if not legal_moves:
        break
    
    # Create the root node from the current state
    root = Decision_Node(state=env.board.copy(), score=env.score)
    td_mcts = TD_MCTS(env, approximator, iterations=50, exploration_constant=0.0, rollout_depth=0, gamma=0.99)

    # Run multiple simulations to build the MCTS tree
    for _ in range(td_mcts.iterations):
        td_mcts.run_simulation(root)

    # Select the best action (based on highest visit count)
    best_act, _ = td_mcts.best_action_distribution(root)
    # print("TD-MCTS selected action:", best_act)

    # Execute the selected action and update the state
    state, reward, done, _ = env.step(best_act)
    # max_tile = np.max(state)
    # if max_tile % 2048 == 0:
    #     print(f"max_tile: {max_tile}")
    # env.render(action=best_act)

env.render()
print("Game over, final score:", env.score)