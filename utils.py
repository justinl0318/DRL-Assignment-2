import numpy as np
import random
from typing import Optional, Union, List, Tuple
from numba import njit
from const import BLACK, WHITE, DIRECTIONS, PLAYER_MAP


def get_candidate_positions(board, player, opponent, is_root, radius: int = 2, k: int = 15, size: int = 19) -> List[Tuple[int, int]]:
    """
    Returns a list of candidate positions for MCTS exploration.
    
    Args:
        radius: Radius around existing stones to consider for moves
        k: Maximum number of candidate positions to return
    
    Returns:
        List of position tuples (row, col)
    """
    # Find all empty positions
    empty_positions = [(r, c) for r in range(size) for c in range(size) 
                      if board[r, c] == 0]
    
    if not empty_positions:
        return []
        
    # Initialize lists for different types of moves
    winning_moves = []
    blocking_moves = []
    threat_moves = []
    strong_positions = []
    
    # Check for immediate winning moves
    for r, c in empty_positions:
        # Check for winning move
        board[r, c] = player
        if _check_win(board, r, c, player):
            if is_root == True:    
                winning_moves.append(((r, c), float("inf")))
            else:
                winning_moves.append(((r, c), 1_000_000_000))
        board[r, c] = 0
        
        # Check for blocking move
        board[r, c] = opponent
        if _check_win(board, r, c, opponent):
            blocking_moves.append(((r, c), float("inf")))
        board[r, c] = 0
    
    # # If we have winning or blocking moves, prioritize those
    # if winning_moves:
    #     if len(winning_moves) == 1:
    #         random_element = (random.choice(empty_positions), float("-inf"))
    #         winning_moves.append(random_element)
    #     return winning_moves 
    
    # if blocking_moves:
    #     if len(blocking_moves) == 1:
    #         random_element = (random.choice(empty_positions), float("-inf"))
    #         blocking_moves.append(random_element)
    #     return blocking_moves
    
    # Find threat patterns
    threats = _find_threat_patterns(board, player)
    threat_moves = [((r, c), val) for r, c, val in threats]
    
    # Evaluate all empty positions
    position_scores = []
    for r, c in empty_positions:
        score = _evaluate_position(board, r, c, player)
        position_scores.append((r, c, score))
    
    # Sort by score (descending)
    position_scores.sort(key=lambda x: x[2], reverse=True)
    
    # Get top k strong positions
    strong_positions = [((r, c), val) for r, c, val in position_scores[:k]]
    
    # Combine all candidate moves, prioritizing threats and strong positions
    candidates = list(dict.fromkeys(threat_moves + strong_positions)) # remove duplicates
    
    # # If we have too few candidates, also consider moves near existing stones
    # if len(candidates) < k:
    #     # Find positions near existing stones
    #     near_stone_positions = set()
    #     for r in range(size):
    #         for c in range(size):
    #             if board[r, c] > 0:  # stone exists
    #                 for dr in range(-radius, radius + 1):
    #                     for dc in range(-radius, radius + 1):
    #                         if dr == 0 and dc == 0:
    #                             continue
    #                         nr, nc = r + dr, c + dc
    #                         if (0 <= nr < size and 0 <= nc < size and 
    #                             board[nr, nc] == 0 and
    #                             (nr, nc) not in near_stone_positions and
    #                             (nr, nc) not in candidates):
    #                             near_stone_positions.add((nr, nc))
        
    #     # Add positions near stones
    #     candidates.extend(list(near_stone_positions))
    
    # Return the top k candidates
    
    if winning_moves:
        return winning_moves + candidates
    elif blocking_moves:
        return blocking_moves + candidates
    else:
        return candidates

@njit
def _check_win(board, r, c, player, size: int = 19):
    """Check if the last move at (r, c) created a winning condition"""
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dr, dc in directions:
        count = 1
        # Check forward
        rr, cc = r + dr, c + dc
        while 0 <= rr < size and 0 <= cc < size and board[rr, cc] == player:
            count += 1
            rr += dr
            cc += dc
        
        # Check backward
        rr, cc = r - dr, c - dc
        while 0 <= rr < size and 0 <= cc < size and board[rr, cc] == player:
            count += 1
            rr -= dr
            cc -= dc
        
        if count >= 6:
            return True
    return False

@njit
def _find_threat_patterns(board, player, size: int = 19):
    """Find threat patterns that could lead to victory"""
    opponent = 3 - player
    threats = []
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    # Look for various threat patterns
    for r in range(size):
        for c in range(size):
            if board[r, c] == player:
                for dr, dc in directions:
                    # Search for patterns like: X X X X _ X or X X X X _ _ where X is player's stone
                    pattern_length = 6
                    stones = []
                    gaps = []
                    
                    for i in range(pattern_length):
                        rr, cc = r + i*dr, c + i*dc
                        if not (0 <= rr < size and 0 <= cc < size):
                            break
                        if board[rr, cc] == player:
                            stones.append((rr, cc))
                        elif board[rr, cc] == 0:
                            gaps.append((rr, cc))
                        else:  # opponent stone
                            break
                    
                    # Check if this is a strong threat pattern
                    if len(stones) >= 4 and len(gaps) <= 2 and len(stones) + len(gaps) >= 6:
                        for gap_r, gap_c in gaps:
                            threats.append((gap_r, gap_c, 1000 * len(stones)))
    
    # Add threat responses for opponent's potential wins
    for r in range(size):
        for c in range(size):
            if board[r, c] == opponent:
                for dr, dc in directions:
                    stones = []
                    gaps = []
                    
                    for i in range(6):
                        rr, cc = r + i*dr, c + i*dc
                        if not (0 <= rr < size and 0 <= cc < size):
                            break
                        if board[rr, cc] == opponent:
                            stones.append((rr, cc))
                        elif board[rr, cc] == 0:
                            gaps.append((rr, cc))
                        else:
                            break
                    
                    if len(stones) >= 4 and len(gaps) <= 2 and len(stones) + len(gaps) >= 6:
                        for gap_r, gap_c in gaps:
                            threats.append((gap_r, gap_c, 2000 * len(stones)))  # Higher priority for defense
    
    return threats

@njit
def _evaluate_position(board, r, c, player, size: int = 19):
    """Evaluate a position's strength"""
    opponent = 3 - player
    score = 0
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    # Place temporary stone
    original_value = board[r, c]
    board[r, c] = player
    
    # Check connections
    for dr, dc in directions:
        count = 1
        open_ends = 0
        
        # Forward check
        rr, cc = r + dr, c + dc
        while 0 <= rr < size and 0 <= cc < size:
            if board[rr, cc] == player:
                count += 1
            elif board[rr, cc] == 0:
                open_ends += 1
                break
            else:
                break
            rr += dr
            cc += dc
        
        # Backward check
        rr, cc = r - dr, c - dc
        while 0 <= rr < size and 0 <= cc < size:
            if board[rr, cc] == player:
                count += 1
            elif board[rr, cc] == 0:
                open_ends += 1
                break
            else:
                break
            rr -= dr
            cc -= dc
        
        # Score based on count and openness
        if count >= 5:
            score += 10000
        elif count == 4 and open_ends == 2:
            score += 5000
        elif count == 4 and open_ends == 1:
            score += 1000
        elif count == 3 and open_ends == 2:
            score += 500
        elif count == 3 and open_ends == 1:
            score += 100
        elif count == 2 and open_ends == 2:
            score += 50
    
    # Check defense value
    board[r, c] = opponent
    for dr, dc in directions:
        count = 1
        open_ends = 0
        
        # Forward
        rr, cc = r + dr, c + dc
        while 0 <= rr < size and 0 <= cc < size:
            if board[rr, cc] == opponent:
                count += 1
            elif board[rr, cc] == 0:
                open_ends += 1
                break
            else:
                break
            rr += dr
            cc += dc
        
        # Backward
        rr, cc = r - dr, c - dc
        while 0 <= rr < size and 0 <= cc < size:
            if board[rr, cc] == opponent:
                count += 1
            elif board[rr, cc] == 0:
                open_ends += 1
                break
            else:
                break
            rr -= dr
            cc -= dc
        
        # Defensive scoring
        if count >= 5:
            score += 9000
        elif count == 4 and open_ends == 2:
            score += 4500
        elif count == 4 and open_ends == 1:
            score += 900
        elif count == 3 and open_ends == 2:
            score += 450
    
    # Center influence scoring
    center = size // 2
    distance_to_center = abs(r - center) + abs(c - center)
    center_bonus = max(0, (size - distance_to_center) * 5)
    score += center_bonus
    
    # Restore the original board state
    board[r, c] = original_value
    
    return score