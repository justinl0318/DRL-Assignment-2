import copy
import random
import math
import numpy as np
from typing import Optional, Union, Tuple, List
from game_env import Game2048Env
from approximator import NTupleApproximator

# Node for TD-MCTS using the TD-trained value approximator
# Chance_Node (ref: https://ko19951231.github.io/2021/01/01/2048/): after a player moves but before the random tile is spawned
class Chance_Node:
    def __init__(self, state, score, parent=None, action=None):
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {} # (pos, value) -> Decision_Node
        self.visits = 0
        self.total_reward = 0.0
        self.all_expanded = False # set to true when the chance node is fully expanded

    def fully_expanded(self):
        return self.all_expanded

# Represent the Player taking a determinstic action
class Decision_Node:
    def __init__(self, state, score, parent=None, action=None):
        """
        state: current board state (numpy array)
        score: cumulative score at this node
        parent: parent node (None for root)
        action: action taken from parent to reach this node
        """
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {} # action -> Chance_Node
        self.visits = 0
        self.total_reward = 0.0

        self.legal_actions = {} # action -> (state_after, state_after_score)
        temp_env = Game2048Env()
        for a in range(4):
            temp_env.board = self.state.copy()
            temp_env.score = score
            state_after, reward, moved, _ = temp_env.step_without_random_tile(a)
            if moved:
                self.legal_actions[a] = (state_after, temp_env.score)

    def fully_expanded(self):
        if not self.legal_actions:
            return False
        for action in self.legal_actions:
            if action not in self.children: 
                return False
        return True
        

# TD-MCTS class utilizing a trained approximator for leaf evaluation
class TD_MCTS:
    def __init__(self, env, approximator, iterations=50, exploration_constant=1.41, rollout_depth=10, gamma=0.99):
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma

    def create_env_from_state(self, state, score) -> Optional[Game2048Env]:
        # Create a deep copy of the environment with the given state and score.
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_decision_child(self, node: Optional[Decision_Node]): # select_child for Decision_Node (select based on UCT formula)
        # TODO: Use the UCT formula: Q + c * sqrt(log(parent.visits)/child.visits) to select the best child.
        best_score = float("-inf")
        best_action = None
        for action, child in node.children.items():
            if child.visits == 0:
                uct_score = self.approximator.value(child.state)
            else:
                Q = child.total_reward / child.visits # average reward
                U = self.c * math.sqrt(math.log(node.visits) / child.visits)
                uct_score = Q + U

            if uct_score > best_score:
                best_action = action
                best_score = uct_score

        return best_action
    
    def select_chance_child(self, node: Optional[Chance_Node]): # select_child for Chance_Node (select based on probability)
        keys = list(node.children.keys()) # possible (pos, val) pairs
        probs = [0.9 if val == 2 else 0.1 for (_, val) in keys]
        sampled_key = random.choices(population=keys, weights=probs, k=1)[0]
        return sampled_key

    def select(self, root: Optional[Union[Decision_Node, Chance_Node]], sim_env: Optional[Game2048Env]): 
        node = root
        rewards = []

        while node.fully_expanded():
            if isinstance(node, Decision_Node):
                action = self.select_decision_child(node)
                state_after, reward, moved, _ = sim_env.step_without_random_tile(action)
                rewards.append(reward)
                node = node.children[action]

            elif isinstance(node, Chance_Node):
                sampled_key = self.select_chance_child(node)
                node = node.children[sampled_key]
                # update sim_env
                sim_env.board = node.state
                sim_env.score = node.score

        return node, rewards
    
    def expand_decision_node_all(self, node: Optional[Decision_Node]):
        for action, (state_after, state_after_score) in node.legal_actions.items():
            chance_node = Chance_Node(state_after.copy(), state_after_score, parent=node, action=action)
            node.children[action] = chance_node
    
    def expand_chance_node_all(self, node: Optional[Chance_Node]):
        empty_tiles = list(zip(*np.where(node.state == 0))) # convert tuples of list to list of tuples
        for row, col in empty_tiles:
            for val in [2, 4]:
                key = ((row, col), val)
                if key in node.children:
                    continue
                new_state = node.state.copy()
                new_state[row, col] = val
                node.children[key] = Decision_Node(new_state, node.score, parent=node, action=key)

        node.all_expanded = True
                
    def expand(self, node, sim_env: Optional[Game2048Env]):
        if sim_env.is_game_over():
            return
        
        if isinstance(node, Decision_Node):
            # if the Decision_Node is first encountered (brand new)
            if not node.children:
                self.expand_decision_node_all(node)

        elif isinstance(node, Chance_Node):
            # expand all of the possibilities of the random tiles   
            self.expand_chance_node_all(node)    

    def evaluate_decision_node(self, sim_env: Optional[Game2048Env], approximator: Optional[NTupleApproximator]):
        # since the approximator was trained on state_after (without random_tile), so we evaluate the current Decision_Node
        # by reward + V(state_after)
        decision_node = Decision_Node(sim_env.board.copy(), sim_env.score)
        if not decision_node.legal_actions:
            return 0
        
        best_value = float("-inf")
        for action, (state_after, state_after_score) in decision_node.legal_actions.items():
            reward = state_after_score - sim_env.score
            value = reward + approximator.value(state_after)
            best_value = max(best_value, value)
        return best_value

    def rollout(self, node, sim_env: Optional[Game2048Env], depth=10):
        value = 0
        if isinstance(node, Decision_Node):
            value = self.evaluate_decision_node(sim_env, self.approximator)
        
        elif isinstance(node, Chance_Node):
            value = self.approximator.value(node.state)

        return value
    
    def backpropagate(self, node, value, rewards_list):
        # We only add value and rewards to Chance_Node, bcuz in select_decision_child(), we select the Chance_Node child using UCT_score;
        # whereas in select_chance_child(), we select the Decision_Node child (random tile) using probability.
        # Suppose it is D1->C1->D2->C2->D3; and node = D3.
        # rewards_list is [r1, r2], where r1 is from D1->C1, and r2 is from D2->C2. 
        # However, r2 should be credited to C1, bcuz r2 is generated from taking the action at D2 (which is the state after adding random tile in C1)
        
        N = len(rewards_list)
        idx = N # purposely start at N instead of N - 1 so that we dont add the reward when we first see a Chance_Node
        cum_value = value
        while node is not None:
            node.visits += 1
            if isinstance(node, Chance_Node):
                if idx < N:
                    cum_value += rewards_list[idx]
                idx -= 1
                node.total_reward += cum_value
            node = node.parent

    def run_simulation(self, root):
        # create a new sim_env for each TD_MCTS simulation
        sim_env = self.create_env_from_state(root.state, root.score)

        node, rewards = self.select(root, sim_env)
        self.expand(node, sim_env)
        value = self.rollout(node, sim_env)
        self.backpropagate(node, value, rewards)

    def best_action_distribution(self, root):
        # Compute the normalized visit count distribution for each child of the root.
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action

        return best_action, distribution